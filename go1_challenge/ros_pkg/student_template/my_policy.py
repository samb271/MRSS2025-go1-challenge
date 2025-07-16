#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Image, LaserScan
import numpy as np
import cv2
from cv_bridge import CvBridge
import math


class StudentPolicyNode(Node):
    """
    Student template for GO1 navigation policy.

    Students should implement their navigation logic in the control_callback method.

    Available inputs:
    - self.robot_pose: Current robot position [x, y] (None if not available)
    - self.robot_yaw: Current robot orientation in radians
    - self.goal: Goal position [x, y]
    - self.current_image: Latest camera image (OpenCV format)
    - self.laser_scan: Latest laser scan data

    Output:
    - Publish Twist message to /cmd_vel with linear.x and angular.z
    """

    def __init__(self):
        super().__init__("student_policy_node")

        # Goal position (can be changed)
        self.goal = np.array([2.0, 2.0])

        # Robot state
        self.robot_pose = None
        self.robot_yaw = 0.0
        self.current_image = None
        self.laser_scan = None

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Subscribers
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/robot_pose", self.pose_callback, 10)

        self.image_sub = self.create_subscription(Image, "/camera/image_raw", self.image_callback, 10)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)

        # Control timer (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_callback)

        # Parameters
        self.goal_tolerance = 0.2
        self.max_linear_vel = 0.5
        self.max_angular_vel = 1.0

        self.get_logger().info("Student policy node initialized")
        self.get_logger().info(f"Goal position: {self.goal}")

    def pose_callback(self, msg):
        """Update robot pose from localization"""
        self.robot_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])

        # Get yaw from quaternion
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def image_callback(self, msg):
        """Update camera image"""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def scan_callback(self, msg):
        """Update laser scan data"""
        self.laser_scan = msg

    def control_callback(self):
        """
        Main control loop - STUDENTS IMPLEMENT THIS!

        This method is called at 10 Hz. Students should implement their
        navigation logic here.
        """

        # Check if we have necessary data
        if self.robot_pose is None:
            self.get_logger().warn("No robot pose available")
            self.publish_zero_velocity()
            return

        # Check if goal is reached
        goal_distance = np.linalg.norm(self.robot_pose - self.goal)
        if goal_distance < self.goal_tolerance:
            self.get_logger().info("Goal reached!")
            self.publish_zero_velocity()
            return

        # =================================================================
        # STUDENT IMPLEMENTATION AREA
        # =================================================================

        # Example simple implementation: move towards goal
        cmd = self.simple_goal_approach()

        # TODO: Replace with your own implementation!
        # You can use:
        # - self.robot_pose: current position [x, y]
        # - self.robot_yaw: current orientation
        # - self.goal: goal position [x, y]
        # - self.current_image: camera image (OpenCV format)
        # - self.laser_scan: laser scan data
        # - self.obstacle_avoidance(): example obstacle avoidance
        # - self.process_image(): example image processing

        # =================================================================

        # Publish command
        self.cmd_pub.publish(cmd)

    def simple_goal_approach(self):
        """
        Simple example: move directly towards goal
        Students should replace this with their own implementation!
        """
        cmd = Twist()

        # Calculate direction to goal
        dx = self.goal[0] - self.robot_pose[0]
        dy = self.goal[1] - self.robot_pose[1]

        # Calculate desired heading
        desired_angle = math.atan2(dy, dx)

        # Calculate angle error
        angle_error = desired_angle - self.robot_yaw

        # Normalize angle
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        # Simple control law
        cmd.linear.x = self.max_linear_vel * (1 - abs(angle_error) / math.pi)
        cmd.angular.z = np.clip(2.0 * angle_error, -self.max_angular_vel, self.max_angular_vel)

        return cmd

    def obstacle_avoidance(self):
        """
        Example obstacle avoidance using laser scan
        Students can use this as a starting point
        """
        if self.laser_scan is None:
            return 0.0  # No angular adjustment

        # Get laser scan ranges
        ranges = np.array(self.laser_scan.ranges)

        # Replace inf values with max range
        ranges[np.isinf(ranges)] = self.laser_scan.range_max

        # Check for close obstacles
        min_distance = 0.5  # meters
        close_obstacles = ranges < min_distance

        if np.any(close_obstacles):
            # Simple avoidance: turn away from obstacles
            angles = np.linspace(self.laser_scan.angle_min, self.laser_scan.angle_max, len(ranges))

            # Calculate repulsive force
            repulsive_angle = 0.0
            for i, (dist, angle) in enumerate(zip(ranges, angles)):
                if dist < min_distance:
                    # Repulsive force inversely proportional to distance
                    force = (min_distance - dist) / min_distance
                    repulsive_angle += force * np.sin(angle + np.pi)  # Opposite direction

            return np.clip(repulsive_angle, -self.max_angular_vel, self.max_angular_vel)

        return 0.0

    def process_image(self):
        """
        Example image processing
        Students can use this for visual navigation
        """
        if self.current_image is None:
            return None

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2HSV)

        # Example: detect red objects (could be goal markers)
        red_lower = np.array([0, 50, 50])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Get centroid
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Return direction to target (-1 left, 0 center, 1 right)
                image_center = self.current_image.shape[1] // 2
                direction = (cx - image_center) / image_center

                return direction

        return None

    def publish_zero_velocity(self):
        """Stop the robot"""
        cmd = Twist()
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = StudentPolicyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
