import torch
import os


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

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, check_file_path, read_file
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.sensors import CameraCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from pxr import UsdGeom, Gf, UsdPhysics

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort:skip

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg import Go1LocomotionEnvCfg_PLAY

##
# Scene
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
class Go1ChallengeSceneCfg(Go1LocomotionEnvCfg_PLAY):
    """Design the scene with Go1 robot and camera."""

    # TODO: camera
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

        # -- Arena
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # TODO: load arena
        # Maybe have three `levels`: 1: Straight line with walls, 2: Arena with obstacles,
        # 3: More obstacles w/ rough terrain
        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="usd",
        #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        # )

        # -- robot
        self.scene.robot.init_state.pos = (1.5, 1.5, 0.4)  # x, y, z position (starting in top-right area)

        self.scene.num_envs = 1

        # -- mdp
        self.episode_length_s = 60.0
        self.curriculum = None

        # TODO: Set to teleop commands
        self.observations.policy.velocity_commands = ObsTerm(func=constant_commands)


##
# MDP settings
##


def constant_commands(env: ManagerBasedEnv) -> torch.Tensor:
    """The generated command from the command generator."""
    return torch.tensor([[1, 0, 0]], device=env.device).repeat(env.num_envs, 1)


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
class FullObservationsCfg:
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
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        velocity_commands = ObsTerm(func=constant_commands)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
