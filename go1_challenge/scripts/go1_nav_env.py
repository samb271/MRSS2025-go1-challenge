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
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go1 navigation environment with ArUco tags.")
parser.add_argument("--disable_fabric", action="store_true", help="Disable Fabric API and use USD instead.")

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

import isaacsim.core.utils.prims as prim_utils


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import Articulation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.sensors import CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaacsim.core.utils.viewports import set_camera_view
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from pxr import UsdGeom, Gf, UsdPhysics

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort:skip

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg_PLAY
from go1_challenge.isaac_sim.go1_challenge_env_cfg import Go1ChallengeSceneCfg


PKG_PATH = Path(__file__).parent.parent.parent

GYM_TASK = "Isaac-Velocity-Flat-Unitree-Go1-v0"
DEVICE = "cpu"


# * OLD - To remove
def load_policy_skrl(policy_file=None):
    """Load the policy for the Go1 robot using skrl agent. NOT WORKING

    Args:
        policy_file (str, optional): Path to the policy file. Relative path to 'logs/skrl/unitree_go1_flat'
    """
    import skrl
    from packaging import version

    # check for minimum supported skrl version
    SKRL_VERSION = "1.4.2"
    if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
        skrl.logger.error(
            f"Unsupported skrl version: {skrl.__version__}. "
            f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
        )
        exit()

    from skrl.utils.runner.torch import Runner
    from isaaclab_tasks.utils import load_cfg_from_registry

    policy_dir = PKG_PATH / "logs" / "skrl" / "unitree_go1_flat"
    policy_path = policy_dir / policy_file

    print(f"[INFO] Loading policy from {policy_path}")

    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    # Load the experiment configuration (same as training)
    try:
        experiment_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-Unitree-Go1-v0", "skrl_ppo_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry("Isaac-Velocity-Flat-Unitree-Go1-v0", "skrl_cfg_entry_point")

    # Create a dummy environment wrapper for the agent (we'll extract the policy later)
    # We need this because skrl agents are tied to environments
    class DummyEnv:
        def __init__(self):
            self.num_envs = 1
            self.num_agents = 1
            self.state_space = type("MockSpace", (), {"shape": (48,)})()  # Adjust based on your observation size
            self.observation_space = self.state_space
            self.action_space = type("MockSpace", (), {"shape": (12,)})()  # Adjust based on your action size
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_env = DummyEnv()

    # env = gym.make(GYM_TASK, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Configure the runner and agent
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(dummy_env, experiment_cfg)

    # Load the checkpoint
    runner.agent.load(str(policy_path))
    runner.agent.set_running_mode("eval")

    # Return the agent's policy network
    return runner.agent


def load_policy_rsl(policy_file=None):
    """Load the policy using rsl-rl format."""
    policy_dir = PKG_PATH / "logs" / "rsl_rl"  # Adjust path as needed
    policy_path: Path = policy_dir / policy_file

    if not policy_path.exists():
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")

    print(f"[INFO] Loading policy from {policy_path}")

    # # Load the policy directly (rsl-rl saves as pickle/torch)
    # checkpoint = torch.load(policy_path, map_location="cpu")

    # t = policy_path.parent / "exported" / "policy.pt"
    file_bytes = read_file(policy_path)
    # jit load the policy
    policy = torch.jit.load(file_bytes)

    return policy
    # # Extract actor network from checkpoint
    # if "model_state_dict" in checkpoint:
    #     policy_state_dict = checkpoint["model_state_dict"]
    # elif "ac_parameters_1" in checkpoint:  # rsl-rl format
    #     policy_state_dict = checkpoint["ac_parameters_1"]
    # else:
    #     policy_state_dict = checkpoint

    # # Create a simple MLP policy network (adjust architecture as needed)
    # class PolicyMLP(torch.nn.Module):
    #     def __init__(self, input_size=48, output_size=12, hidden_dims=[512, 256, 128]):
    #         super().__init__()
    #         layers = []
    #         prev_dim = input_size

    #         for hidden_dim in hidden_dims:
    #             layers.extend([torch.nn.Linear(prev_dim, hidden_dim), torch.nn.ELU()])
    #             prev_dim = hidden_dim

    #         layers.append(torch.nn.Linear(prev_dim, output_size))
    #         self.network = torch.nn.Sequential(*layers)

    #     def forward(self, x):
    #         return self.network(x)

    # policy = PolicyMLP()

    # # Load state dict (may need filtering for actor-only weights)
    # try:
    #     policy.load_state_dict(policy_state_dict)
    # except RuntimeError:
    #     # Filter for actor weights only if full AC model is saved
    #     actor_weights = {k.replace("actor.", ""): v for k, v in policy_state_dict.items() if "actor" in k}
    #     policy.load_state_dict(actor_weights)

    # return policy


def create_aruco_map_yaml():
    """Create ArUco map YAML file with tag positions"""
    tag_positions = [
        (2.4, 2.4, 0.3, 0, (0.0, 0.0, -0.707, 0.707)),  # ID 0: Top-right corner
        (-2.4, 2.4, 0.3, 1, (0.0, 0.0, 0.707, 0.707)),  # ID 1: Top-left corner
        (-2.4, -2.4, 0.3, 2, (0.0, 0.0, 1.0, 0.0)),  # ID 2: Bottom-left corner
        (2.4, -2.4, 0.3, 3, (0.0, 0.0, 0.0, 1.0)),  # ID 3: Bottom-right corner
        (0.0, 2.4, 0.3, 4, (0.0, 0.0, 1.0, 0.0)),  # ID 4: Top wall center
        (0.0, -2.4, 0.3, 5, (0.0, 0.0, 0.0, 1.0)),  # ID 5: Bottom wall center
    ]

    tag_map = {}
    for x, y, z, tag_id, rot in tag_positions:
        tag_map[tag_id] = {
            "position": [float(x), float(y), float(z)],
            "orientation": [float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])],  # quaternion
            "size": 0.15,
        }

    # Save tag map YAML
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, "aruco_map.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump({"tags": tag_map}, f, default_flow_style=False)
    print(f"Generated tag map: {yaml_path}")


def load_gym_env() -> ManagerBasedRLEnv:
    """Load the Gym environment."""
    # #! RL Env
    # env_cfg = Go1LocomotionEnvCfg_PLAY()
    # env_cfg.scene.num_envs = 1
    # env_cfg.curriculum = None
    # # env_cfg.scene.terrain = TerrainImporterCfg(
    # #     prim_path="/World/ground",
    # #     terrain_type="usd",
    # #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    # # )
    # env_cfg.sim.device = DEVICE  # args_cli.device
    # if DEVICE == "cpu":
    #     env_cfg.sim.use_fabric = False
    # env = ManagerBasedRLEnv(cfg=env_cfg)

    # #! Challenge Env
    env_cfg = Go1ChallengeSceneCfg()
    env_cfg.sim.device = DEVICE  # args_cli.device
    if DEVICE == "cpu":
        env_cfg.sim.use_fabric = False
    env = ManagerBasedRLEnv(cfg=env_cfg)

    return env


def main():
    """Main function."""
    # # Generate ArUco tag textures if they don't exist
    # texture_dir = os.path.join(os.path.dirname(__file__), "textures", "aruco_tags")
    # if not os.path.exists(texture_dir) or len(os.listdir(texture_dir)) < 6:
    #     print("Generating ArUco tag textures...")
    #     from generate_aruco_textures import generate_aruco_tags

    #     generate_aruco_tags()

    # # Create ArUco map YAML file
    # create_aruco_map_yaml()

    # setup environment
    env = load_gym_env()  # To load env used for training

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    policy = load_policy_rsl("unitree_go1_rough/2025-07-17_13-20-43/exported/policy.pt")  # Adjust filename
    # policy = load_policy_rsl("unitree_go1_flat/2025-07-16_17-10-21/exported/policy.pt")
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
