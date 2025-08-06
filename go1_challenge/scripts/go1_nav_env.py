# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates Go1 robot navigation in an arena with ArUco tags.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/go1_nav_env.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 navigation environment with ArUco tags.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")
parser.add_argument(
    "--level",
    type=int,
    default=2,
    choices=[1, 2, 3],
    help="Challenge level: 1=flat no obstacles, 2=flat with obstacles, 3=rough with obstacles",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import yaml
import gymnasium as gym


import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaacsim.core.utils.viewports import set_camera_view

##
# Pre-defined configs
##
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg_PLAY
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_challenge_env_cfg import (
    create_go1_challenge_env_cfg,
    constant_commands,
)


PKG_PATH = Path(__file__).parent.parent.parent
DEVICE = "cpu"


def load_policy_rsl(policy_file=None):
    """Load the policy using rsl-rl format."""
    policy_dir = PKG_PATH / "logs" / "rsl_rl"  # Adjust path as needed
    policy_path: Path = policy_dir / policy_file

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    print(f"[INFO] Loading policy from {policy_path}")

    # Load the policy directly (rsl-rl saves as pickle/torch)
    file_bytes = read_file(policy_path)

    # jit load the policy
    policy = torch.jit.load(file_bytes)

    return policy


def load_gym_env() -> ManagerBasedRLEnv:
    """Load the Gym environment."""
    env_cfg = create_go1_challenge_env_cfg(level=args_cli.level)
    env_cfg.sim.device = DEVICE
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False

    # Set constant commands for autonomous policy inference
    from isaaclab.managers import ObservationTermCfg as ObsTerm

    env_cfg.observations.policy.velocity_commands = ObsTerm(func=constant_commands)

    # Disable camera for this example
    env_cfg.scene.camera = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def main():
    """Main function."""
    # setup environment
    env = load_gym_env()

    print(f"[INFO] Challenge Level {args_cli.level} loaded")

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    policy = load_policy_rsl("unitree_go1_rough/2025-07-17_18-39-56/exported/policy.pt")  # Adjust filename
    policy = policy.to(env.device).eval()

    # Simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 100 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # infer action directly from policy network
            action = policy(obs["policy"])

            # step env
            obs, _, _, _, _ = env.step(action)

            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    # close sim app
    simulation_app.close()
