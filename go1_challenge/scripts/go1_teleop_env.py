# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates Go1 robot teleoperation using keyboard to set velocity commands
for a trained RSL-RL policy in an arena with ArUco tags.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p go1_challenge/scripts/go1_teleop_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 teleoperation with trained policy and keyboard velocity commands.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--policy",
    type=str,
    default="unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt",
    help="Path to the trained policy file.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import numpy as np
import torch
import carb
import cv2

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import read_file
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.devices import Se2Keyboard

# AprilTag detection
try:
    import apriltag

    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False
    print("Warning: apriltag library not available. Install with: pip install apriltag")

##
# Pre-defined configs
##
from go1_challenge.isaac_sim.go1_challenge_env_cfg import Go1ChallengeSceneCfg

PKG_PATH = Path(__file__).parent.parent.parent
DEVICE = "cpu"

NAV_FREQ = 4
LOCALIZATION_FREQ = 10  # Frequency for localization updates


def load_policy_rsl(policy_file: str):
    """Load the trained RSL-RL policy."""
    policy_dir = PKG_PATH / "logs" / "rsl_rl"  # Adjust path as needed
    policy_path: Path = policy_dir / policy_file
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    print(f"[INFO] Loading policy from {policy_path}")

    file_bytes = read_file(policy_path)
    policy = torch.jit.load(file_bytes)

    return policy


def load_gym_env() -> ManagerBasedRLEnv:
    """Load the Gym environment with keyboard velocity commands."""
    env_cfg = Go1ChallengeSceneCfg()
    env_cfg.sim.device = DEVICE
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False

    env_cfg.episode_length_s = 60.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    simulation_app.close()


def get_camera_intrinsics(camera_cfg):
    """Extract camera intrinsic parameters from IsaacLab camera configuration."""
    # Camera parameters from CameraCfg
    focal_length = camera_cfg.spawn.focal_length  # in mm
    horizontal_aperture = camera_cfg.spawn.horizontal_aperture  # in mm
    width = camera_cfg.width
    height = camera_cfg.height

    # Convert focal length to pixels
    # fx = fy = (focal_length / horizontal_aperture) * width
    fx = fy = (focal_length * width) / horizontal_aperture

    # Principal point (assuming centered)
    cx = width / 2.0
    cy = height / 2.0

    # Camera intrinsic matrix
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    # Distortion coefficients (assuming no distortion for sim)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    return camera_matrix, dist_coeffs


def detect_apriltags(
    rgb_image: torch.Tensor, camera_matrix: np.ndarray, tag_size: float = 0.2, tag_family: str = "tag36h11"
) -> dict:
    """
    Detect AprilTags in RGB image and estimate their poses relative to camera frame.

    Args:
        rgb_image: RGB image tensor from IsaacLab camera (H, W, 3)
        camera_matrix: 3x3 camera intrinsic matrix
        tag_size: Physical size of tags in meters
        tag_family: AprilTag family name

    Returns:
        Dictionary of detected tags with poses in robot/camera frame
    """
    if not APRILTAG_AVAILABLE:
        print("AprilTag library not available")
        return {}

    try:
        # Convert tensor to numpy and ensure correct format
        if isinstance(rgb_image, torch.Tensor):
            # Convert from tensor to numpy, ensure CPU and correct dtype
            image_np = rgb_image.detach().cpu().numpy()
        else:
            image_np = rgb_image

        # Ensure image is in uint8 format and correct shape
        if image_np.dtype != np.uint8:
            # Assuming image is in [0, 1] range, convert to [0, 255]
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)

        # Convert to grayscale for AprilTag detection
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Initialize AprilTag detector
        detector = apriltag.Detector(apriltag.DetectorOptions(families=tag_family))

        # Detect tags
        results = detector.detect(gray)

        if len(results) == 0:
            return {}

        detected_tags = {}

        # Camera parameters for pose estimation
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        for detection in results:
            tag_id = detection.tag_id

            # Get corner points
            corners = detection.corners.astype(np.float32)
            center = detection.center

            # 3D object points for tag (tag lies in XY plane, Z=0)
            # Standard AprilTag coordinate system: origin at center, edges at ±tag_size/2
            half_size = tag_size / 2.0
            object_points = np.array(
                [
                    [-half_size, -half_size, 0],  # Bottom-left
                    [half_size, -half_size, 0],  # Bottom-right
                    [half_size, half_size, 0],  # Top-right
                    [-half_size, half_size, 0],  # Top-left
                ],
                dtype=np.float32,
            )

            # Solve PnP to get pose
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corners,
                camera_matrix,
                np.zeros((4, 1)),  # No distortion in simulation
            )

            if success:
                # Convert rotation vector to rotation matrix
                # rotation_matrix, _ = cv2.Rodrigues(rvec)

                # Convert rotation matrix to quaternion (w, x, y, z)
                def rotation_matrix_to_quaternion(R):
                    """Convert rotation matrix to quaternion."""
                    trace = np.trace(R)
                    if trace > 0:
                        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                        qw = 0.25 * s
                        qx = (R[2, 1] - R[1, 2]) / s
                        qy = (R[0, 2] - R[2, 0]) / s
                        qz = (R[1, 0] - R[0, 1]) / s
                    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
                        qw = (R[2, 1] - R[1, 2]) / s
                        qx = 0.25 * s
                        qy = (R[0, 1] + R[1, 0]) / s
                        qz = (R[0, 2] + R[2, 0]) / s
                    elif R[1, 1] > R[2, 2]:
                        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
                        qw = (R[0, 2] - R[2, 0]) / s
                        qx = (R[0, 1] + R[1, 0]) / s
                        qy = 0.25 * s
                        qz = (R[1, 2] + R[2, 1]) / s
                    else:
                        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
                        qw = (R[1, 0] - R[0, 1]) / s
                        qx = (R[0, 2] + R[2, 0]) / s
                        qy = (R[1, 2] + R[2, 1]) / s
                        qz = 0.25 * s
                    return [qw, qx, qy, qz]

                # quaternion = rotation_matrix_to_quaternion(rotation_matrix)

                # Calculate distance
                distance = np.linalg.norm(tvec)

                detected_tags[tag_id] = {
                    "corners": corners.tolist(),
                    "center": center.tolist(),
                    "pose": {
                        "position": tvec.flatten().tolist(),  # [x, y, z] in camera frame
                        # "rotation": quaternion,  # [qw, qx, qy, qz]
                        # "rotation_matrix": rotation_matrix.tolist(),
                    },
                    "distance": float(distance),
                    "confidence": float(detection.decision_margin),  # Detection confidence
                }

        print(f"[INFO] Detected {len(detected_tags)} AprilTags: {list(detected_tags.keys())}")
        return detected_tags

    except Exception as e:
        print(f"[ERROR] AprilTag detection failed: {e}")
        return {}


def main():
    """Main teleoperation loop."""

    # Setup environment
    env = load_gym_env()

    # Setup keyboard interface
    # keyboard_interface = VelocityKeyboardInterface(lin_vel_scale=0.8, ang_vel_scale=0.6)
    sensitivity_lin = 1.0  # Default sensitivity for keyboard commands
    sensitivity_ang = 1.0  # Default sensitivity for angular commands
    teleop_interface = Se2Keyboard(
        v_x_sensitivity=sensitivity_lin, v_y_sensitivity=sensitivity_lin, omega_z_sensitivity=sensitivity_ang
    )

    teleop_interface.add_callback("R", env.reset)
    teleop_interface.add_callback("ESCAPE", quit_cb)

    print(teleop_interface)

    # Load trained policy
    policy = load_policy_rsl("unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt")  # Adjust filename
    policy = policy.to(env.device).eval()

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Reset environment
    teleop_interface.reset()
    obs, _ = env.reset()

    count = 0
    nav_count = 1 / (NAV_FREQ * env.cfg.sim.dt)
    localization_count = 1 / (LOCALIZATION_FREQ * env.cfg.sim.dt)

    # Get camera intrinsics from environment configuration
    camera_cfg = env.cfg.scene.camera
    camera_matrix, dist_coeffs = get_camera_intrinsics(camera_cfg)

    print(f"[INFO] Camera intrinsics calculated:")
    print(f"  Camera matrix:\n{camera_matrix}")
    print(f"  Image size: {camera_cfg.width}x{camera_cfg.height}")

    print("\n[INFO] Teleoperation started. Use WASD+QE to control the robot.")

    while simulation_app.is_running():
        with torch.inference_mode():
            # Navigation loop
            if count % localization_count == 0:
                # Get camera images
                rgb_data = env.scene["camera"].data.output["rgb"][0, ..., :3]
                distance_data = env.scene["camera"].data.output["distance_to_image_plane"][0]

                # AprilTag localization
                detected_tags = detect_apriltags(
                    rgb_image=rgb_data,
                    camera_matrix=camera_matrix,
                    tag_size=0.2,  # 0.2m tag size as specified
                    tag_family="tag36h11",
                )

                if detected_tags:
                    print("-------------------------------")
                    print(f"[LOCALIZATION] Detected tags: {list(detected_tags.keys())}")
                    for tag_id, tag_data in detected_tags.items():
                        pos = tag_data["pose"]["position"]
                        dist = tag_data["distance"]
                        print(f"  Tag {tag_id}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), dist={dist:.2f}m")

                    print("-------------------------------")

                # print("-------------------------------")
                # print("Received shape of rgb   image: ", rgb_data.shape)
                # print("Received shape of depth image: ", distance_data.shape)
                # print("-------------------------------")

            # Teleop Commands
            keyboard_command = teleop_interface.advance()
            obs["policy"][:, 9:12] = torch.tensor(keyboard_command)  # Update policy observation with keyboard command

            # Policy inference
            action = policy(obs["policy"])

            # Step environment
            obs, _, _, _, _ = env.step(action)

            count += 1

    # Cleanup
    # keyboard_interface.cleanup()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
