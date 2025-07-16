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

import isaacsim.core.utils.prims as prim_utils


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import Articulation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
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

from pxr import UsdGeom, Gf, UsdPhysics

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort:skip

PKG_PATH = Path(__file__).parent.parent.parent.parent

##
# Custom observation terms
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


##
# Helper functions for scene generation
##


def create_arena_walls(size=5.0, wall_height=0.4, wall_thickness=0.2):
    """Create arena wall configurations"""
    walls = {}

    # Wall positions and sizes
    wall_configs = [
        ("wall_right", (2.5, 0.0, 0.2), (wall_thickness, size, wall_height)),
        ("wall_left", (-2.5, 0.0, 0.2), (wall_thickness, size, wall_height)),
        ("wall_front", (0.0, 2.5, 0.2), (size, wall_thickness, wall_height)),
        ("wall_back", (0.0, -2.5, 0.2), (size, wall_thickness, wall_height)),
    ]

    for i, (name, pos, size_tuple) in enumerate(wall_configs):
        walls[name] = AssetBaseCfg(
            prim_path=f"/World/Wall_{i}",
            spawn=sim_utils.CuboidCfg(
                size=size_tuple,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        )

    return walls


def create_aruco_tag_configs(tag_size=0.15, post_height=0.3):
    """Create ArUco tag and post configurations"""
    # Tag positions on walls - positioned flat against walls
    tag_positions = [
        # Corner tags on walls
        (2.4, 2.4, 0.3, 0, (0.0, 0.0, -0.707, 0.707)),  # ID 0: Top-right corner, facing inward
        (-2.4, 2.4, 0.3, 1, (0.0, 0.0, 0.707, 0.707)),  # ID 1: Top-left corner, facing inward
        (-2.4, -2.4, 0.3, 2, (0.0, 0.0, 1.0, 0.0)),  # ID 2: Bottom-left corner, facing inward
        (2.4, -2.4, 0.3, 3, (0.0, 0.0, 0.0, 1.0)),  # ID 3: Bottom-right corner, facing inward
        # Mid-wall tags
        (0.0, 2.4, 0.3, 4, (0.0, 0.0, 1.0, 0.0)),  # ID 4: Top wall center, facing inward
        (0.0, -2.4, 0.3, 5, (0.0, 0.0, 0.0, 1.0)),  # ID 5: Bottom wall center, facing inward
    ]

    configs = {}

    for x, y, z, tag_id, rot in tag_positions:
        # Get the texture path
        texture_path = os.path.join(os.path.dirname(__file__), "textures", "aruco_tags", f"aruco_tag_{tag_id}.png")

        # Create ArUco tag on wall (no posts needed)
        configs[f"aruco_tag_{tag_id}"] = AssetBaseCfg(
            prim_path=f"/World/ArUco_{tag_id}",
            spawn=sim_utils.CuboidCfg(
                size=(tag_size, tag_size, 0.001),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_texture=texture_path if os.path.exists(texture_path) else None,
                    diffuse_color=(1.0, 1.0, 1.0),  # White background if no texture
                    roughness=0.1,
                    metallic=0.0,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z), rot=rot),
        )

    return configs


@configclass
class Go1NavSceneCfg(InteractiveSceneCfg):
    """Design the scene with Go1 robot and camera."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg(size=(5.0, 5.0)))

    # lights
    light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 1.5, 0.4),  # x, y, z position (starting in top-right area)
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Central obstacle
    obstacle = AssetBaseCfg(
        prim_path="/World/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.5,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )

    # # camera
    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.1), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
    # )

    def __post_init__(self):
        """Post initialization to add dynamically generated components."""
        super().__post_init__()

        # Add arena walls
        wall_configs = create_arena_walls()
        for name, config in wall_configs.items():
            setattr(self, name, config)

        # Add ArUco tags (no posts needed since they're on walls)
        # tag_configs = create_aruco_tag_configs()
        # for name, config in tag_configs.items():
        #     setattr(self, name, config)


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: Go1NavSceneCfg = Go1NavSceneCfg(num_envs=1, env_spacing=2.5)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device

        # self.sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz


def load_policy(policy_path=None):
    """Load the policy for the Go1 robot.

    Args:
        policy_path (str, optional): Path to the policy file. Relative path to 'logs/skrl/unitree_go1_flat'

    """
    # load level policy
    # policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    policy_dir = PKG_PATH / "logs" / "skrl" / "unitree_go1_flat"
    policy_path = policy_dir / policy_path

    print(f"[INFO] Loading policy from {policy_path}")

    # check if policy file exists
    if not check_file_path(policy_path):
        raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    file_bytes = read_file(policy_path)

    # jit load the policy
    policy = torch.jit.load(file_bytes)

    return policy


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


def run_simulator(env: ManagerBasedEnv):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = env.sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # Create output directory to save images
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # policy
    policy = load_policy("2025-07-15_12-48-05_ppo_torch/checkpoints/best_agent.pt")
    policy = policy.to(env.device).eval()

    # Simulate physics
    obs, _ = env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            # Reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")

            # # Apply default actions to the robot
            # targets = scene["robot"].data.default_joint_pos
            # scene["robot"].set_joint_position_target(targets)
            # scene.write_data_to_sim()

            # # perform step
            # sim.step()
            # # update sim-time
            # sim_time += sim_dt
            # count += 1
            # # update buffers
            # scene.update(sim_dt)

            # infer action
            action = policy(obs["policy"])

            # # step env
            obs, _ = env.step(action)

            # update counter
            count += 1

            # # print camera information
            # if count % 50 == 0:
            #     print("-------------------------------")
            #     # print(scene["camera"])
            #     # print("Received shape of rgb image: ", scene["camera"].data.output["rgb"].shape)


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

    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # # Initialize the simulation context
    # sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    # sim = sim_utils.SimulationContext(sim_cfg)

    # # design scene
    # design_scene()

    # # create interactive scene
    # scene_cfg = Go1NavSceneCfg(num_envs=1, env_spacing=2.0)
    # scene = InteractiveScene(scene_cfg)

    # # Play the simulator
    # env.reset()
    # # Now we are ready!
    # print("[INFO]: Setup complete...")
    # # Run the simulator
    # run_simulator(env)

    # simulate physics

    set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            # action = policy(obs["policy"])
            action = torch.zeros((env.num_envs, 12), device=env.device)
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
