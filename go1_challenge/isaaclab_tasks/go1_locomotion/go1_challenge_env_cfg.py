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
import isaaclab.terrains as terrain_gen

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

    # Class attribute to store the level
    level: int = 1

    def __post_init__(self):
        """Post initialization to add dynamically generated components."""
        super().__post_init__()

        # Import arena
        arena_usd_path = ARENA_ASSETS_DIR / "arena_5x5.usd"
        if not arena_usd_path.exists():
            raise FileNotFoundError(f"Arena USD file not found: {arena_usd_path}")

        # Add arena walls and tags as separate assets
        self.scene.arena_walls = AssetBaseCfg(
            prim_path="/World/arena_structure",
            spawn=sim_utils.UsdFileCfg(
                usd_path=arena_usd_path.as_posix(),
                # Only spawn the wall/tag structures, not the ground plane
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            collision_group=-1,
        )

        # Configure ground based on level
        if self.level in [1, 2]:
            # Level 1 & 2: Flat arena with AprilTags
            arena_terrain_cfg = TerrainGeneratorCfg(
                size=(2.5, 2.5),  # Slightly larger than arena to cover edges
                border_width=0.1,
                num_rows=2,
                num_cols=2,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
                },
            )

        else:
            # Level 3: Arena with rough terrain overlay
            arena_terrain_cfg = TerrainGeneratorCfg(
                size=(2.5, 2.5),  # Slightly larger than arena to cover edges
                border_width=0.02,
                num_rows=2,
                num_cols=2,
                horizontal_scale=0.1,
                vertical_scale=0.005,
                slope_threshold=0.75,
                use_cache=False,
                sub_terrains={
                    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                        proportion=0.5, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
                    ),
                    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                        proportion=0.5, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25
                    ),
                    # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.0),
                },
            )

        # Use multi-terrain approach: arena base + rough heightfield
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=arena_terrain_cfg,
            max_init_terrain_level=0,  # Single terrain level
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # -- Camera (all levels)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk/front_cam",
            update_period=0.1,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.25, 0.0, 0.08), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )

        # -- Dynamic obstacles (Level 2 & 3 only)
        if self.level >= 2:
            self.scene.obstacle_1 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_1",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_15"},
                ),
                collision_group=-1,
            )

            self.scene.obstacle_2 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_2",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_16"},
                ),
                collision_group=-1,
            )

            self.scene.obstacle_3 = AssetBaseCfg(
                prim_path="/World/obstacles/obstacle_3",
                init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, -1.0, 0.0)),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=(ARENA_ASSETS_DIR / "assets" / "obstacle.usd").as_posix(),
                    scale=(1.0, 1.0, 1),
                    variants={"Tag": "_16"},
                ),
                collision_group=-1,
            )

            # Add obstacle randomization event
            self.events.randomize_obstacle_positions = EventTerm(
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

        # -- Robot spawn configuration
        self.scene.robot.init_state.pos = (0, 0, 0.4)
        self.scene.num_envs = 1

        # -- MDP settings
        self.episode_length_s = 60.0
        self.curriculum = None
        self.observations.policy.velocity_commands = ObsTerm(func=constant_commands)

        # Update robot spawn to be in bottom left corner safe zone
        # self.events.reset_base.params["pose_range"] = {"x": (-1.25, -1.25), "y": (-1.25, -1.25), "yaw": (-3.14, 3.14)}
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)}

        print(f"[INFO] Go1 Challenge Level {self.level} configured:")
        if self.level == 1:
            print("  - Flat arena with AprilTags")
            print("  - No obstacles")

        elif self.level == 2:
            print("  - Flat arena with AprilTags")
            print("  - 3 dynamic obstacles")

        elif self.level == 3:
            print("  - Arena walls and AprilTags")
            print("  - Rough terrain heightfield")
            print("  - 3 dynamic obstacles")


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
