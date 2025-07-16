#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf2_geometry_msgs
import numpy as np
import cv2
from cv_bridge import CvBridge
import yaml
import os
from scipy.spatial.transform import Rotation


class ArUcoLocalizationNode(Node):
    def __init__(self):
        super().__init__("aruco_localization_node")

        # Parameters
        self.declare_parameter("tag_map_file", "config/aruco_map.yaml")
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("dictionary_id", cv2.aruco.DICT_4X4_50)

        # Load ArUco tag map
        self.tag_map = self.load_tag_map()

        # CV Bridge
        self.bridge = CvBridge()

        # ArUco detector
        self.aruco_dict = cv2.aruco.Dictionary_get(self.get_parameter("dictionary_id").value)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Camera matrix (should be loaded from calibration)
        self.camera_matrix = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.zeros(5)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/robot_pose", 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, self.get_parameter("camera_topic").value, self.image_callback, 10
        )

        # Localization state
        self.last_pose = None
        self.pose_confidence = 0.0

        self.get_logger().info("ArUco localization node initialized")

    def load_tag_map(self):
        """Load ArUco tag map from YAML file"""
        tag_map_file = self.get_parameter("tag_map_file").value

        try:
            with open(tag_map_file, "r") as f:
                data = yaml.safe_load(f)
                return data["tags"]
        except FileNotFoundError:
            self.get_logger().error(f"Tag map file not found: {tag_map_file}")
            return {}

    def image_callback(self, msg):
        """Process camera image for ArUco detection"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Detect ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None and len(ids) > 0:
                # Process detected markers
                robot_poses = []

                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in self.tag_map:
                        # Get marker pose in camera frame
                        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                            corners[i : i + 1], self.tag_map[marker_id]["size"], self.camera_matrix, self.dist_coeffs
                        )

                        # Calculate robot pose from marker detection
                        robot_pose = self.calculate_robot_pose(marker_id, rvec[0], tvec[0])

                        if robot_pose is not None:
                            robot_poses.append(robot_pose)

                # Fuse multiple detections
                if robot_poses:
                    final_pose = self.fuse_pose_estimates(robot_poses)
                    self.publish_robot_pose(final_pose, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def calculate_robot_pose(self, marker_id, rvec, tvec):
        """Calculate robot pose from marker detection"""
        # Get marker world position
        marker_world_pos = np.array(self.tag_map[marker_id]["position"])
        marker_world_quat = np.array(self.tag_map[marker_id]["orientation"])

        # Convert camera detection to transformation
        R_cam_marker, _ = cv2.Rodrigues(rvec)

        # Transform from camera frame to robot frame
        # This assumes camera is mounted on robot with known offset
        camera_to_robot_offset = np.array([0.1, 0.0, 0.0])  # 10cm forward

        # Camera pose relative to marker
        R_marker_cam = R_cam_marker.T
        t_marker_cam = -R_marker_cam @ tvec.flatten()

        # Robot pose relative to marker
        t_marker_robot = t_marker_cam + camera_to_robot_offset

        # Convert to world frame
        marker_world_R = Rotation.from_quat(marker_world_quat).as_matrix()

        robot_world_pos = marker_world_pos + marker_world_R @ t_marker_robot
        robot_world_R = marker_world_R @ R_marker_cam

        robot_world_quat = Rotation.from_matrix(robot_world_R).as_quat()

        return {
            "position": robot_world_pos,
            "orientation": robot_world_quat,
            "confidence": 1.0 / (np.linalg.norm(tvec) + 0.1),  # Closer = more confident
        }

    def fuse_pose_estimates(self, poses):
        """Fuse multiple pose estimates using weighted average"""
        if len(poses) == 1:
            return poses[0]

        # Weighted average by confidence
        total_weight = sum(p["confidence"] for p in poses)

        # Average position
        avg_pos = np.zeros(3)
        for pose in poses:
            weight = pose["confidence"] / total_weight
            avg_pos += weight * pose["position"]

        # Average orientation (simplified - should use proper quaternion averaging)
        avg_quat = np.zeros(4)
        for pose in poses:
            weight = pose["confidence"] / total_weight
            avg_quat += weight * pose["orientation"]
        avg_quat = avg_quat / np.linalg.norm(avg_quat)

        return {"position": avg_pos, "orientation": avg_quat, "confidence": total_weight / len(poses)}

    def publish_robot_pose(self, pose, timestamp):
        """Publish robot pose estimate"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = "world"

        # Position
        pose_msg.pose.pose.position.x = pose["position"][0]
        pose_msg.pose.pose.position.y = pose["position"][1]
        pose_msg.pose.pose.position.z = pose["position"][2]

        # Orientation
        pose_msg.pose.pose.orientation.x = pose["orientation"][0]
        pose_msg.pose.pose.orientation.y = pose["orientation"][1]
        pose_msg.pose.pose.orientation.z = pose["orientation"][2]
        pose_msg.pose.pose.orientation.w = pose["orientation"][3]

        # Covariance (simplified)
        confidence = pose["confidence"]
        pose_uncertainty = 1.0 / (confidence + 0.1)

        pose_msg.pose.covariance[0] = pose_uncertainty  # x
        pose_msg.pose.covariance[7] = pose_uncertainty  # y
        pose_msg.pose.covariance[35] = pose_uncertainty * 0.1  # yaw

        self.pose_pub.publish(pose_msg)
        self.last_pose = pose


def main(args=None):
    rclpy.init(args=args)
    node = ArUcoLocalizationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
