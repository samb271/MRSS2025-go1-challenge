#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, OccupancyGrid
import numpy as np
import heapq
from collections import defaultdict
import math


class BaselinePlannerNode(Node):
    def __init__(self):
        super().__init__("baseline_planner_node")

        # Parameters
        self.declare_parameter("goal_x", 2.0)
        self.declare_parameter("goal_y", 2.0)
        self.declare_parameter("grid_resolution", 0.05)
        self.declare_parameter("arena_size", 5.0)

        # Goal
        self.goal = np.array([self.get_parameter("goal_x").value, self.get_parameter("goal_y").value])

        # Grid parameters
        self.resolution = self.get_parameter("grid_resolution").value
        self.arena_size = self.get_parameter("arena_size").value
        self.grid_size = int(self.arena_size / self.resolution)

        # Occupancy grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.setup_static_obstacles()

        # Robot state
        self.robot_pose = None
        self.current_path = []
        self.path_index = 0

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/planned_path", 10)

        # Subscribers
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/robot_pose", self.pose_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Control timer
        self.control_timer = self.create_timer(0.1, self.control_callback)  # 10 Hz

        # Planning parameters
        self.goal_tolerance = 0.2
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0
        self.lookahead_distance = 0.3

        self.get_logger().info("Baseline planner initialized")

    def setup_static_obstacles(self):
        """Setup static obstacles in the grid"""
        center = self.grid_size // 2

        # Central obstacle (cylinder with radius 0.25m)
        obstacle_radius_cells = int(0.3 / self.resolution)  # Slightly larger for safety

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Distance from center
                dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
                if dist < obstacle_radius_cells:
                    self.grid[i, j] = 100  # Occupied

        # Walls
        wall_thickness = int(0.2 / self.resolution)

        # Top and bottom walls
        self.grid[:wall_thickness, :] = 100
        self.grid[-wall_thickness:, :] = 100

        # Left and right walls
        self.grid[:, :wall_thickness] = 100
        self.grid[:, -wall_thickness:] = 100

    def pose_callback(self, msg):
        """Update robot pose"""
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Get yaw from quaternion
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        # Replan if needed
        if self.should_replan():
            self.plan_path()

    def scan_callback(self, msg):
        """Update dynamic obstacles from laser scan"""
        if self.robot_pose is None:
            return

        # Clear previous dynamic obstacles
        self.grid[self.grid == 50] = 0  # Clear dynamic obstacles

        # Add new obstacles from scan
        for i, range_val in enumerate(msg.ranges):
            if range_val < msg.range_max and range_val > msg.range_min:
                angle = msg.angle_min + i * msg.angle_increment

                # Convert to world coordinates
                world_angle = self.robot_yaw + angle
                obstacle_x = self.robot_pose[0] + range_val * np.cos(world_angle)
                obstacle_y = self.robot_pose[1] + range_val * np.sin(world_angle)

                # Convert to grid coordinates
                grid_x, grid_y = self.world_to_grid(obstacle_x, obstacle_y)

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # Inflate obstacle slightly
                    inflation_radius = int(0.2 / self.resolution)
                    for dx in range(-inflation_radius, inflation_radius + 1):
                        for dy in range(-inflation_radius, inflation_radius + 1):
                            gx, gy = grid_x + dx, grid_y + dy
                            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                                if self.grid[gx, gy] == 0:  # Don't overwrite static obstacles
                                    self.grid[gx, gy] = 50  # Dynamic obstacle

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x + self.arena_size / 2) / self.resolution)
        grid_y = int((y + self.arena_size / 2) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        x = (grid_x * self.resolution) - self.arena_size / 2
        y = (grid_y * self.resolution) - self.arena_size / 2
        return x, y

    def should_replan(self):
        """Check if replanning is needed"""
        if not self.current_path:
            return True

        # Check if path is blocked
        for i in range(min(len(self.current_path), 20)):  # Check next 20 waypoints
            if i >= len(self.current_path):
                break

            waypoint = self.current_path[i]
            grid_x, grid_y = self.world_to_grid(waypoint[0], waypoint[1])

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size and self.grid[grid_x, grid_y] > 0:
                return True

        return False

    def plan_path(self):
        """Plan path using A* algorithm"""
        if self.robot_pose is None:
            return

        # Convert to grid coordinates
        start_grid = self.world_to_grid(self.robot_pose[0], self.robot_pose[1])
        goal_grid = self.world_to_grid(self.goal[0], self.goal[1])

        # A* search
        path = self.astar(start_grid, goal_grid)

        if path:
            # Convert back to world coordinates
            self.current_path = [self.grid_to_world(gx, gy) for gx, gy in path]
            self.path_index = 0

            # Publish path for visualization
            self.publish_path()

            self.get_logger().info(f"Planned path with {len(self.current_path)} waypoints")
        else:
            self.get_logger().warn("No path found to goal")
            self.current_path = []

    def astar(self, start, goal):
        """A* pathfinding algorithm"""

        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = defaultdict(lambda: float("inf"))
        g_score[start] = 0

        f_score = defaultdict(lambda: float("inf"))
        f_score[start] = heuristic(start, goal)

        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue

                # Check if occupied
                if self.grid[neighbor[0], neighbor[1]] > 0:
                    continue

                # Calculate tentative g_score
                move_cost = np.sqrt(dx**2 + dy**2)
                tentative_g_score = g_score[current] + move_cost

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def control_callback(self):
        """Main control loop"""
        if self.robot_pose is None or not self.current_path:
            return

        # Check if goal reached
        goal_distance = np.linalg.norm(self.robot_pose - self.goal)
        if goal_distance < self.goal_tolerance:
            self.publish_zero_velocity()
            self.get_logger().info("Goal reached!")
            return

        # Pure pursuit control
        cmd = self.pure_pursuit_control()
        self.cmd_pub.publish(cmd)

    def pure_pursuit_control(self):
        """Pure pursuit path following"""
        cmd = Twist()

        # Find lookahead point
        lookahead_point = self.find_lookahead_point()

        if lookahead_point is None:
            self.publish_zero_velocity()
            return cmd

        # Calculate heading error
        dx = lookahead_point[0] - self.robot_pose[0]
        dy = lookahead_point[1] - self.robot_pose[1]

        target_angle = math.atan2(dy, dx)
        angle_error = target_angle - self.robot_yaw

        # Normalize angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Control law
        cmd.linear.x = self.max_linear_vel * (1 - abs(angle_error) / math.pi)
        cmd.angular.z = np.clip(2.0 * angle_error, -self.max_angular_vel, self.max_angular_vel)

        return cmd

    def find_lookahead_point(self):
        """Find lookahead point on path"""
        if not self.current_path:
            return None

        # Start from current path index
        for i in range(self.path_index, len(self.current_path)):
            waypoint = self.current_path[i]
            distance = np.linalg.norm(waypoint - self.robot_pose)

            if distance >= self.lookahead_distance:
                self.path_index = i
                return waypoint

        # Return last waypoint if no lookahead point found
        return self.current_path[-1]

    def publish_zero_velocity(self):
        """Publish zero velocity command"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)

    def publish_path(self):
        """Publish planned path for visualization"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"

        for waypoint in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = waypoint[0]
            pose.pose.position.y = waypoint[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BaselinePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
