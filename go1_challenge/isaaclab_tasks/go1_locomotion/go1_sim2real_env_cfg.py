"""
Definition of the Go1 Sim2Real environment configuration for IsaacSim. This environment is designed to be used
during the sim2real tutorial.
"""

import isaaclab.envs.mdp as mdp
from isaaclab.terrains import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen

from isaaclab.utils import configclass


##
# Pre-defined configs
##
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg

SIM2REAL_TERRAIN = TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(2.0, 2.0),
    border_width=0.0,
    num_rows=5,
    num_cols=3,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.33, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=0.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.33, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.0
        ),
        # "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.33),
    },
)
"""Sim2Real terrain: flat, boxes and rough"""


@configclass
class Go1ChallengeSceneCfg(Go1LocomotionEnvCfg):
    def __post_init__(self):
        """Post initialization to add dynamically generated components."""
        super().__post_init__()

        self.seed = 42  # Set a fixed seed for reproducibility

        # Set camera view
        self.viewer.eye = (2.0, 0.0, 5.0)
        self.viewer.lookat = (-3.0, 0.0, 0.0)

        # make a smaller scene for play
        self.scene.num_envs = 3
        self.curriculum.terrain_levels = None  # Disable terrain levels increasing

        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

        self.scene.terrain.terrain_generator = SIM2REAL_TERRAIN
        self.scene.terrain.max_init_terrain_level = 0

        # Fix velocity command ranges for Go1
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
        )

        # Push
        self.events.push_robot.interval_range_s = [3.0, 3.0]  # Push every 5 seconds
        self.events.push_robot.params["velocity_range"] = {"x": (-0.0, -0.0), "y": (0.0, 0.0)}

        added_mass = 3.0
        self.events.add_base_mass.params["mass_distribution_params"] = (added_mass, added_mass)
