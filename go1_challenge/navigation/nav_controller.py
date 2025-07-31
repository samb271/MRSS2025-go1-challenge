"""
Navigation Controller base class for Go1 Challenge.

This class handles robot localization and navigation command generation.
Students will implement the core methods to complete the navigation system.
"""

import numpy as np
import torch
import cv2
import time
import os
from typing import Dict, Tuple, Optional, Any

# AprilTag detection
try:
    from pyapriltags import Detector

    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("Warning: pyapriltags library not available. Install with: pip install pyapriltags")


class NavController:
    """
    Base Navigation Controller for Go1 robot navigation with AprilTag localization.

    The controller receives observations at regular intervals and maintains internal
    state for robot pose estimation and navigation planning.
    """

    def __init__(
        self, camera_params: Tuple[float, float, float, float], tag_size: float = 0.16, tag_family: str = "tag36h11"
    ):
        """
        Initialize the navigation controller.

        Args:
            camera_params: Camera intrinsics (fx, fy, cx, cy) for AprilTag detection
            tag_size: Physical size of AprilTags in meters (length of black square)
            tag_family: AprilTag family name (e.g., "tag36h11")

        Students should implement their state initialization here, including:
        - Robot pose estimates (position, orientation)
        - Map or landmark storage
        - Kalman filter/particle filter initialization
        - Any other navigation-related state
        """
        # Camera and AprilTag parameters
        self.camera_params = camera_params
        self.tag_size = tag_size
        self.tag_family = tag_family

        # Initialize AprilTag detector
        if APRILTAG_AVAILABLE:
            self.at_detector = Detector(
                families=tag_family,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
        else:
            self.at_detector = None
            print("[WARNING] AprilTag detection not available")

        # Example state variables (students can modify/extend):
        self.robot_pose = np.array([0.0, 0.0, 0.0])  # [x, y, yaw] in world frame
        self.pose_covariance = np.eye(3) * 0.1  # Pose uncertainty
        self.landmark_map = {}  # Dict to store landmark positions
        self.last_update_time = 0.0

        # Students can add more state variables as needed
        pass

    def detect_apriltags(self, rgb_image: torch.Tensor, visualize: bool = False) -> Dict[int, Dict]:
        """
        Detect AprilTags in RGB image and estimate their poses.

        Students can override this method to implement their own detection logic
        or use the raw images for other processing.

        Args:
            rgb_image: RGB image tensor from camera (H, W, 3)
            visualize: Whether to save visualization images to disk

        Returns:
            Dictionary of detected tags with poses in camera frame
            Format: {tag_id: {"pose": {"position": [x,y,z], "rotation_matrix": [[...]]},
                             "distance": float, "confidence": float}}
        """
        if not APRILTAG_AVAILABLE or self.at_detector is None:
            print("[WARNING] AprilTag detection not available")
            return {}

        try:
            # Convert tensor to numpy and ensure correct format
            if isinstance(rgb_image, torch.Tensor):
                image_np = rgb_image.detach().cpu().numpy()
            else:
                image_np = rgb_image

            # Ensure image is in uint8 format
            if image_np.dtype != np.uint8:
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                else:
                    image_np = image_np.astype(np.uint8)

            # Convert to grayscale for AprilTag detection
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Detect tags with pose estimation
            tags = self.at_detector.detect(
                gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=self.tag_size
            )

            if len(tags) == 0:
                if visualize:
                    self._save_visualization_image(image_np, [], "no_tags")
                return {}

            detected_tags = {}

            # Process detected tags
            for tag in tags:
                tag_id = tag.tag_id
                position = tag.pose_t.flatten()  # Translation vector [x, y, z]
                rotation_matrix = tag.pose_R  # 3x3 rotation matrix
                distance = np.linalg.norm(position)

                detected_tags[tag_id] = {
                    "corners": tag.corners.tolist(),
                    "center": tag.center.tolist(),
                    "pose": {
                        "position": position.tolist(),  # [x, y, z] in camera frame
                        "rotation_matrix": rotation_matrix.tolist(),
                    },
                    "distance": float(distance),
                    "confidence": float(tag.decision_margin),
                }

            # Save visualization if requested
            if visualize:
                self._save_visualization_image(
                    image_np, tags, f"tags_{'_'.join(map(str, sorted(detected_tags.keys())))}"
                )

            return detected_tags

        except Exception as e:
            print(f"[ERROR] AprilTag detection failed: {e}")
            return {}

    def _save_visualization_image(self, image_np: np.ndarray, tags: list, prefix: str):
        """Save visualization image with detected tags to disk."""
        try:
            # Convert grayscale to color if needed
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                vis_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                vis_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

            # Draw detected tags
            for tag in tags:
                # Draw tag outline
                corners = tag.corners.astype(int)
                for idx in range(len(corners)):
                    cv2.line(vis_image, tuple(corners[idx - 1]), tuple(corners[idx]), (0, 255, 0), 2)

                # Add tag ID text
                center = tag.center.astype(int)
                cv2.putText(
                    vis_image,
                    f"ID:{tag.tag_id}",
                    org=(center[0] - 20, center[1] - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=(0, 0, 255),
                    thickness=2,
                )

                # Add distance text
                distance = np.linalg.norm(tag.pose_t)
                cv2.putText(
                    vis_image,
                    f"{distance:.2f}m",
                    org=(center[0] - 20, center[1] + 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 0, 0),
                    thickness=1,
                )

            # Save image
            timestamp = int(time.time() * 1000)
            save_dir = "/tmp/apriltag_detection"
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{save_dir}/{prefix}_{timestamp}.jpg"

            # Convert RGB to BGR for OpenCV saving
            vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, vis_image_bgr)

            print(f"[INFO] Saved visualization to {filename}")

        except Exception as e:
            print(f"[WARNING] Failed to save visualization: {e}")

    def update(self, observations: Dict[str, Any]) -> None:
        """
        Update internal navigation state based on sensor observations.

        This method is called at regular intervals (10Hz) with the latest observations.
        Students should implement localization and planning logic here.

        Args:
            observations: Dictionary containing:
                - 'base_ang_vel': Angular velocity [rad/s] (3,) array [roll_rate, pitch_rate, yaw_rate]
                - 'projected_gravity': Gravity vector in body frame (3,) array
                - 'velocity_commands': Current velocity commands (3,) array [vx, vy, wz]
                - 'joint_pos': Joint positions (12,) array
                - 'joint_vel': Joint velocities (12,) array
                - 'actions': Last action taken (12,) array
                - 'rgb_image': RGB camera image tensor (H, W, 3) or None

        Note:
            - rgb_image may be None if camera is not available
            - Students should detect AprilTags from rgb_image using self.detect_apriltags()
            - All other observations are in appropriate coordinate frames (body/world)
        """
        # Get RGB image and detect AprilTags
        rgb_image = observations.get("rgb_image", None)
        detected_tags = {}

        if rgb_image is not None:
            # Students can customize visualization and detection
            detected_tags = self.detect_apriltags(rgb_image, visualize=False)

        # Get other sensor data
        base_ang_vel = observations.get("base_ang_vel", np.zeros(3))
        projected_gravity = observations.get("projected_gravity", np.zeros(3))

        # Students implement localization logic here:
        # 1. Dead reckoning using IMU/odometry
        # 2. AprilTag-based localization updates using detected_tags
        # 3. Sensor fusion (EKF, particle filter, etc.)
        # 4. Map/landmark management

        # Example: Print detected tags for debugging
        if detected_tags:
            print(f"[NavController] Detected tags: {list(detected_tags.keys())}")

        # TODO: Implement student localization logic
        pass

    def get_velocity_command(self) -> Tuple[float, float, float]:
        """
        Generate velocity command based on current navigation state.

        This method is called at control frequency to get the desired robot motion.
        Students should implement navigation/path planning logic here.

        Returns:
            Tuple of (lin_vel_x, lin_vel_y, ang_vel_z) in robot body frame.
            All values should be in range [-1, 1] representing normalized velocities.
        """
        # Students implement navigation command generation here:
        # 1. Path planning to goal/waypoints
        # 2. Obstacle avoidance
        # 3. PID/MPC control for trajectory following
        # 4. Behavior planning (exploration, goal seeking, etc.)

        # Example placeholder - simple forward motion:
        lin_vel_x = 0.5  # Move forward at half speed
        lin_vel_y = 0.0  # No lateral motion
        ang_vel_z = 0.0  # No rotation

        # TODO: Implement student navigation logic

        # Ensure commands are in valid range
        lin_vel_x = np.clip(lin_vel_x, -1.0, 1.0)
        lin_vel_y = np.clip(lin_vel_y, -1.0, 1.0)
        ang_vel_z = np.clip(ang_vel_z, -1.0, 1.0)

        return lin_vel_x, lin_vel_y, ang_vel_z

    def reset(self) -> None:
        """
        Reset the navigation controller state.

        Called when the environment/robot is reset.
        """
        # Students implement reset logic here
        self.robot_pose = np.array([0.0, 0.0, 0.0])
        self.pose_covariance = np.eye(3) * 0.1
        self.landmark_map.clear()

        # TODO: Implement student reset logic
        pass

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for visualization/logging."""
        return {
            "robot_pose": self.robot_pose.tolist(),
            "pose_covariance": self.pose_covariance.tolist(),
            "landmark_count": len(self.landmark_map),
            "landmark_map": self.landmark_map,
            "camera_params": self.camera_params,
            "tag_size": self.tag_size,
            "tag_family": self.tag_family,
        }
