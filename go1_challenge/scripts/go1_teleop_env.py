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

# Remove AprilTag detection import
# from go1_challenge.isaac_sim.worlds.tags_loc import TAGS_LOC

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 teleoperation with trained policy and keyboard velocity commands.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--policy",
    type=str,
    default="unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt",
    help="Path to the trained policy file.",
)
parser.add_argument("--visualize_tags", action="store_true", help="Visualize detected AprilTags in a separate window.")
parser.add_argument(
    "--teleop",
    action="store_true",
    help="Use keyboard teleoperation. If false, use NavController for autonomous navigation.",
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
import time
import os

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import read_file
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.devices import Se2Keyboard

from go1_challenge.navigation import NavController

##
# Pre-defined configs
##
from go1_challenge.isaac_sim.go1_challenge_env_cfg import Go1ChallengeSceneCfg

PKG_PATH = Path(__file__).parent.parent.parent
DEVICE = "cpu"

NAV_FREQ = 4
LOCALIZATION_FREQ = 10  # Frequency for localization updates


def load_policy_rsl(policy_file: str):
    """Load the trained RSL-RL policy.

    Args:
        policy_file (str): Path to the policy file. Relative to the project root.
    """
    policy_dir = PKG_PATH  # / "logs" / "rsl_rl"  # Adjust path as needed
    policy_path: Path = policy_dir / policy_file
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    print(f"[INFO] Loading policy from {policy_path}")

    file_bytes = read_file(policy_path)
    try:
        policy = torch.jit.load(file_bytes)

    except RuntimeError as e:
        raise RuntimeError(f"Failed to load policy from {policy_path}: {e}. Did you properly export the policy?")

    return policy


def load_gym_env() -> ManagerBasedRLEnv:
    """Load the Gym environment with keyboard velocity commands."""
    env_cfg = Go1ChallengeSceneCfg()
    env_cfg.sim.device = DEVICE
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False

    env_cfg.episode_length_s = 60.0

    env_cfg.observations.policy.base_lin_vel = None  # Use null to disable this observation.

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
    fx = fy = (focal_length * width) / horizontal_aperture

    # Principal point (assuming centered)
    cx = width / 2.0
    cy = height / 2.0

    # Camera parameters tuple format for NavController: (fx, fy, cx, cy)
    camera_params = (fx, fy, cx, cy)

    return camera_params


def get_observation_dict(obs: dict) -> dict:
    """Convert observation tensor to a dictionary for easier access."""
    obs_dict = {
        "base_lin_vel": obs["policy"][:, 0:3].cpu().numpy().flatten(),
        "base_ang_vel": obs["policy"][:, 3:6].cpu().numpy().flatten(),
        "projected_gravity": obs["policy"][:, 6:9].cpu().numpy().flatten(),
        "velocity_commands": obs["policy"][:, 9:12].cpu().numpy().flatten(),
        "joint_pos": obs["policy"][:, 12:24].cpu().numpy().flatten(),
        "joint_vel": obs["policy"][:, 24:36].cpu().numpy().flatten(),
        "actions": obs["policy"][:, 36:48].cpu().numpy().flatten(),
    }
    return obs_dict


def main():
    """Main teleoperation loop."""

    # Setup environment
    env = load_gym_env()

    # --- Setup keyboard interface (only if teleoperation is enabled)
    if args_cli.teleop:
        sensitivity_lin = 1.0
        sensitivity_ang = 1.0
        teleop_interface = Se2Keyboard(
            v_x_sensitivity=sensitivity_lin, v_y_sensitivity=sensitivity_lin, omega_z_sensitivity=sensitivity_ang
        )

        teleop_interface.add_callback("R", env.reset)
        teleop_interface.add_callback("ESCAPE", quit_cb)

        print(teleop_interface)
        print("\n[INFO] Teleoperation mode enabled. Use WASD+QE to control the robot.")

    else:
        teleop_interface = None
        print("\n[INFO] Autonomous navigation mode enabled. NavController will control the robot.")

    # --- Load trained policy
    policy_file = args_cli.policy
    policy = load_policy_rsl(policy_file)
    policy = policy.to(env.device).eval()

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # --- Reset environment
    teleop_interface.reset()
    obs, _ = env.reset()

    count = 0
    nav_count = 1 / (NAV_FREQ * env.cfg.sim.dt)
    localization_count = 1 / (LOCALIZATION_FREQ * env.cfg.sim.dt)

    # --- Initialize NavController
    camera_cfg = env.cfg.scene.camera
    camera_params = get_camera_intrinsics(camera_cfg)
    nav_controller = NavController(
        camera_params=camera_params,
        tag_size=0.16,  # Length of the black square in meters
        tag_family="tag36h11",
    )

    print(f"[INFO] NavController initialized:")
    print(f"  Camera params: {camera_params}")
    print(f"  Tag size: 0.16m")
    print(f"  Tag family: tag36h11")

    # --- Main loop
    while simulation_app.is_running():
        with torch.inference_mode():
            # Navigation loop - update NavController
            if count % localization_count == 0:
                # --- Get camera data
                rgb_data = env.scene["camera"].data.output["rgb"][0, ..., :3]
                distance_data = env.scene["camera"].data.output["distance_to_image_plane"][0]

                obs_dict = get_observation_dict(obs)

                # Prepare observations for NavController
                nav_observations = {
                    "base_ang_vel": obs["policy"][:, 3:6].cpu().numpy().flatten(),
                    "projected_gravity": obs["policy"][:, 6:9].cpu().numpy().flatten(),
                    "velocity_commands": obs["policy"][:, 9:12].cpu().numpy().flatten(),
                    "joint_pos": obs["policy"][:, 12:24].cpu().numpy().flatten(),
                    "joint_vel": obs["policy"][:, 24:36].cpu().numpy().flatten(),
                    "actions": obs["policy"][:, 36:48].cpu().numpy().flatten(),
                    "rgb_image": env.scene["camera"].data.output["rgb"][0, ..., :3]
                    if env.scene.get("camera")
                    else None,
                }

                # Update NavController with observations
                nav_controller.update(nav_observations)

            # Get velocity command based on mode
            if args_cli.teleop and teleop_interface is not None:
                # Use keyboard commands for teleoperation
                command = teleop_interface.advance()
                # print(f"[TELEOP] Keyboard command: {command}")

            else:
                # Use NavController commands for autonomous navigation
                command = nav_controller.get_velocity_command()
                print(f"[NAV] NavController command: {command}")

            # Update policy observation with velocity command
            obs["policy"][:, 6:9] = torch.tensor(command)

            # Policy inference
            action = policy(obs["policy"])

            # Step environment
            obs, _, _, _, _ = env.step(action)

            count += 1

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
