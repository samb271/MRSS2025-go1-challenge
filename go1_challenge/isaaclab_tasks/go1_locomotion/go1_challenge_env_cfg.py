"""
Definition of the Go1 Challenge environment configuration for IsaacSim. This environment is designed to test the Go1
locomotion policy in the arena with obstacles and ArUco tags.
"""

from pathlib import Path
import torch
import os
import numpy as np


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.assets import Articulation

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.sensors import CameraCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.devices import Se2Keyboard

from pxr import UsdGeom, Gf, UsdPhysics, Sdf
import omni.usd

from isaacsim.core.prims import XFormPrim

##
# Pre-defined configs
##
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg_PLAY

##
# Scene
##
# import go1_challenge.arena_assets

ARENA_ASSETS_DIR = Path(__file__).parent.parent.parent / "arena_assets"


@configclass
class Go1ChallengeSceneCfg(Go1LocomotionEnvCfg_PLAY):
    """Design the scene with Go1 robot and camera."""

    def __post_init__(self):
        """Post initialization to add dynamically generated components."""
        super().__post_init__()

        # -- Arena: Use custom USD file
        arena_usd_path = ARENA_ASSETS_DIR / "arena_5x5.usd"

        if not arena_usd_path.exists():
            raise FileNotFoundError(f"Arena USD file not found: {arena_usd_path}")

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=arena_usd_path.as_posix(),  # Ensure path is in POSIX format for USD
        )

        # -- Camera
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.25, 0.0, 0.08), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
            ),  # TODO: Validate
        )

        # -- Dynamic obstacles
        # Prism obstacle 1
        self.scene.obstacle_1 = AssetBaseCfg(
            prim_path="/World/obstacles/obstacle_1",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 1.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                # variants="v1",
                scale=(1.0, 1.0, 1),
                #     rigid_props=RigidBodyPropertiesCfg(
                #         solver_position_iteration_count=16,
                #         solver_velocity_iteration_count=1,
                #         max_angular_velocity=1000.0,
                #         max_linear_velocity=1000.0,
                #         max_depenetration_velocity=5.0,
                #         disable_gravity=False,
                #     ),
                # ),
                variants={
                    "Tag": "_15",
                },
            ),
        )

        self.scene.obstacle_2 = AssetBaseCfg(
            prim_path="/World/obstacles/obstacle_2",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                scale=(1.0, 1.0, 1),
                variants={
                    "Tag": "_16",
                },
            ),
        )

        self.scene.obstacle_3 = AssetBaseCfg(
            prim_path="/World/obstacles/obstacle_3",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                scale=(1.0, 1.0, 1),
                variants={
                    "Tag": "_16",
                },
            ),
        )

        # -- robot: Random spawn in safe zone (center 0.5x0.5m area)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)

        self.scene.num_envs = 1

        # -- mdp
        self.episode_length_s = 60.0
        self.curriculum = None

        self.observations.policy.velocity_commands = ObsTerm(func=constant_commands)

        # -- Add event for obstacle randomization
        self.events.randomize_obstacles = EventTerm(
            func=randomize_obstacle_positions,
            mode="reset",
            params={
                "asset_cfgs": [
                    SceneEntityCfg("obstacle_1"),
                    SceneEntityCfg("obstacle_2"),
                    SceneEntityCfg("obstacle_3"),
                ],
                "arena_size": 5.0,
                "wall_buffer": 1.0,
                "obstacle_buffer": 1.5,
                "safe_zone_size": 2.5,
            },
        )

        # Update robot spawn to be random within safe zone
        self.events.reset_base.params["pose_range"] = {"x": (-1.25, -1.25), "y": (-1.25, -1.25), "yaw": (-3.14, 3.14)}


##
# MDP settings
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


def randomize_obstacle_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: SceneEntityCfg,
    arena_size,
    wall_buffer,
    obstacle_buffer,
    safe_zone_size,
):
    """Randomly position obstacles in the arena avoiding walls and safe zone."""

    stage = omni.usd.get_context().get_stage()

    # Define boundaries
    min_coord = -arena_size / 2 + wall_buffer
    max_coord = arena_size / 2 - wall_buffer

    # # Robot spawns in bottom left corner, safe zone extends from that corner
    # robot_spawn_x = -arena_size / 2 + 0.25  # Bottom left corner with small offset
    # robot_spawn_y = -arena_size / 2 + 0.25  # Bottom left corner with small offset

    # Safe zone coordinates - from robot spawn position extending inward by safe_zone_size
    safe_zone_min_x = -arena_size / 2  # robot_spawn_x - safe_zone_size
    safe_zone_max_x = -arena_size / 2 + safe_zone_size  # robot_spawn_x + 0.25  # Small buffer beyond spawn point
    safe_zone_min_y = -arena_size / 2  # robot_spawn_y - 0.25  # Small buffer beyond spawn point
    safe_zone_max_y = -arena_size / 2 + safe_zone_size  # robot_spawn_y + 0.25  # Small buffer beyond spawn point

    positions = []

    # Find valid positions for each asset
    for _ in asset_cfgs:
        max_attempts = 100
        for _ in range(max_attempts):
            # Random position within arena bounds
            x = np.random.uniform(min_coord, max_coord)
            y = np.random.uniform(min_coord, max_coord)

            # Check if position is outside safe zone (robot spawn area)
            in_safe_zone = (safe_zone_min_x <= x <= safe_zone_max_x) and (safe_zone_min_y <= y <= safe_zone_max_y)
            if in_safe_zone:
                continue  # Too close to robot spawn area

            # Check distance from other obstacles
            valid_position = True
            for prev_pos in positions:
                distance = np.sqrt((x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2)
                if distance < obstacle_buffer:
                    valid_position = False
                    break

            if valid_position:
                positions.append((x, y))
                break
        else:
            # Fallback to safe position if no valid position found
            angle = len(positions) * 2 * np.pi / len(asset_cfgs)
            x = 1.5 * np.cos(angle)
            y = 1.5 * np.sin(angle)
            positions.append((x, y))

    # Apply positions to assets
    for i, asset_cfg in enumerate(asset_cfgs):
        x, y = positions[i]
        z = 0.0  # Height for all obstacles

        # Get asset from scene
        asset: XFormPrim = env.scene[asset_cfg.name]

        #
        prim_paths = asset.prim_paths

        for _, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id]

            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            translate_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:translate")

            translate_spec.default = Gf.Vec3f(*[x, y, z])

        # # Set new position
        # new_pos = torch.tensor([x, y, z], device=env.device).unsqueeze(0)
        # asset.write_data_to_sim(torch.cat([new_pos, torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)], dim=1))


def keyboard_velocity_commands(env: ManagerBasedEnv, teleop_device: Se2Keyboard) -> torch.Tensor:
    """Get velocity commands from keyboard input."""
    # This function should be implemented to read keyboard inputs
    # For now, we return a constant command
    return torch.tensor([[0.0, 0.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
