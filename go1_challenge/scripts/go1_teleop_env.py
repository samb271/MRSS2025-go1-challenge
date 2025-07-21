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

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch
import carb

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import read_file
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.devices import Se2Keyboard

##
# Pre-defined configs
##
from go1_challenge.isaac_sim.go1_challenge_env_cfg import Go1ChallengeSceneCfg, keyboard_velocity_commands

PKG_PATH = Path(__file__).parent.parent.parent
DEVICE = "cpu"


class VelocityKeyboardInterface:
    """Keyboard interface for setting velocity commands."""

    def __init__(self, lin_vel_scale: float = 1.0, ang_vel_scale: float = 1.0):
        self.lin_vel_scale = lin_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.velocity_command = torch.zeros(3)  # [lin_vel_x, lin_vel_y, ang_vel_z]
        self.pressed_keys = set()

        # Key mappings
        self.key_bindings = {
            "w": (0, self.lin_vel_scale),  # forward
            "s": (0, -self.lin_vel_scale),  # backward
            "a": (1, self.lin_vel_scale),  # left
            "d": (1, -self.lin_vel_scale),  # right
            "q": (2, self.ang_vel_scale),  # rotate left
            "e": (2, -self.ang_vel_scale),  # rotate right
        }

        self._setup_keyboard()

        print("Velocity Control Keys:")
        print("  W/S: Forward/Backward")
        print("  A/D: Left/Right")
        print("  Q/E: Rotate Left/Right")
        print("  R: Reset environment")
        print("  ESC: Exit")

    def _setup_keyboard(self):
        """Setup keyboard event handling."""
        import omni.appwindow

        self.app_window = omni.appwindow.get_default_app_window()
        self.input_interface = carb.input.acquire_input_interface()
        self.keyboard = self.app_window.get_keyboard()

        # Subscribe to keyboard events
        self.sub_keyboard = self.input_interface.subscribe_to_keyboard_events(self.keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event):
        """Handle keyboard events."""
        if event.input >= 128:  # Skip non-ASCII keys
            return

        key_name = chr(event.input).lower()

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed_keys.add(key_name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed_keys.discard(key_name)

        # Update velocity commands based on pressed keys
        self.velocity_command.zero_()
        for key in self.pressed_keys:
            if key in self.key_bindings:
                axis, value = self.key_bindings[key]
                self.velocity_command[axis] += value

    def get_command(self) -> torch.Tensor:
        """Get current velocity command."""
        return self.velocity_command.clone()

    def should_reset(self) -> bool:
        """Check if reset key was pressed."""
        return "r" in self.pressed_keys

    def should_exit(self) -> bool:
        """Check if exit key was pressed."""
        return chr(27) in [chr(event.input) for event in []]  # ESC key handling would be more complex

    def cleanup(self):
        """Cleanup keyboard subscription."""
        if hasattr(self, "sub_keyboard"):
            self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self.sub_keyboard)


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

    # # Use keyboard velocity commands for observations
    # env_cfg.observations.policy.velocity_commands = ObsTerm(
    #     func=keyboard_velocity_commands, params={"teleop_device": teleop_interface}
    # )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    return env


def quit_cb():
    """Dummy callback function executed when the key 'ESC' is pressed."""
    print("Quit callback")
    simulation_app.close()


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

    print("\n[INFO] Teleoperation started. Use WASD+QE to control the robot.")

    while simulation_app.is_running():
        with torch.inference_mode():
            # # Check for reset
            # if keyboard_interface.should_reset() or count % 1000 == 0:
            #     obs, _ = env.reset()
            #     keyboard_interface.pressed_keys.discard("r")  # Clear reset key
            #     count = 0
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")

            # # Get velocity command from keyboard
            # velocity_cmd = keyboard_interface.get_command()

            # # Set keyboard commands on environment for observation function
            # env._keyboard_commands = velocity_cmd.unsqueeze(0)  # Add batch dimension

            keyboard_command = teleop_interface.advance()
            # print(keyboard_command)
            obs["policy"][:, 9:12] = torch.tensor(keyboard_command)  # Update policy observation with keyboard command

            # Use trained policy to get joint position actions
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
