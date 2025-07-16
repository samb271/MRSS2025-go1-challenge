from pathlib import Path
from isaacsim import SimulationApp

kit = SimulationApp()

import omni.usd
from pxr import UsdGeom, Gf, UsdPhysics, UsdShade
import yaml
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.prims import create_prim
# from isaacsim.core.materials import PhysicsMaterial


class ArenaGenerator:
    def __init__(self, stage_path="/World"):
        self.stage = omni.usd.get_context().get_stage()
        self.stage_path = stage_path

    def create_arena(self, size=5.0, wall_height=0.4, wall_thickness=0.2):
        """Create the main arena with walls"""
        # Create ground plane
        ground_path = f"{self.stage_path}/Ground"
        ground_prim = create_prim(ground_path, "Cube")

        # Set ground dimensions
        cube_geom = UsdGeom.Cube(ground_prim)
        cube_geom.CreateSizeAttr(1.0)

        # Scale to arena size
        xform = UsdGeom.Xformable(ground_prim)
        xform_ops = xform.GetOrderedXformOps()
        if len(xform_ops) >= 3:  # translate, orient, scale
            xform_ops[2].Set(Gf.Vec3d(size, size, 0.1))  # scale
            xform_ops[0].Set(Gf.Vec3d(0, 0, -0.05))  # translate
        else:
            xform.AddScaleOp().Set(Gf.Vec3d(size, size, 0.1))
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))

        # Add physics
        UsdPhysics.CollisionAPI.Apply(ground_prim)

        # Create boundary walls
        wall_positions = [
            (size / 2, 0, wall_height / 2),  # Right wall
            (-size / 2, 0, wall_height / 2),  # Left wall
            (0, size / 2, wall_height / 2),  # Front wall
            (0, -size / 2, wall_height / 2),  # Back wall
        ]

        wall_scales = [
            (wall_thickness, size, wall_height),
            (wall_thickness, size, wall_height),
            (size, wall_thickness, wall_height),
            (size, wall_thickness, wall_height),
        ]

        for i, (pos, scale) in enumerate(zip(wall_positions, wall_scales)):
            wall_path = f"{self.stage_path}/walls/Wall_{i}"
            wall_prim = create_prim(wall_path, "Cube")

            wall_geom = UsdGeom.Cube(wall_prim)
            wall_geom.CreateSizeAttr(1.0)

            xform = UsdGeom.Xformable(wall_prim)
            xform_ops = xform.GetOrderedXformOps()
            if len(xform_ops) >= 3:  # translate, orient, scale
                xform_ops[2].Set(Gf.Vec3d(*scale))  # scale
                xform_ops[0].Set(Gf.Vec3d(*pos))  # translate
            else:
                xform.AddScaleOp().Set(Gf.Vec3d(*scale))
                xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

            UsdPhysics.CollisionAPI.Apply(wall_prim)

    def create_central_obstacle(self, radius=1, height=0.5):
        """Create central cylindrical obstacle"""
        obstacle_path = f"{self.stage_path}/Obstacle"
        obstacle_prim = create_prim(obstacle_path, "Cylinder")

        cylinder_geom = UsdGeom.Cylinder(obstacle_prim)
        cylinder_geom.CreateRadiusAttr(radius)
        cylinder_geom.CreateHeightAttr(height)

        xform = UsdGeom.Xformable(obstacle_prim)
        xform_ops = xform.GetOrderedXformOps()
        if len(xform_ops) >= 1:  # translate
            xform_ops[0].Set(Gf.Vec3d(0, 0, height / 2))
        else:
            xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, height / 2))

        UsdPhysics.CollisionAPI.Apply(obstacle_prim)

    def create_aruco_tags(self, tag_size=0.15, post_height=0.3):
        """Create ArUco tags on posts around the arena"""
        # Tag positions around the arena perimeter
        tag_positions = [
            # Corners
            (2.2, 2.2, post_height, 0),  # ID 0: Top-right
            (-2.2, 2.2, post_height, 1),  # ID 1: Top-left
            (-2.2, -2.2, post_height, 2),  # ID 2: Bottom-left
            (2.2, -2.2, post_height, 3),  # ID 3: Bottom-right
            # Mid-edges
            (0, 2.2, post_height, 4),  # ID 4: Top-center
            (0, -2.2, post_height, 5),  # ID 5: Bottom-center
        ]

        tag_map = {}

        for i, (x, y, z, tag_id) in enumerate(tag_positions):
            # Create post
            post_path = f"{self.stage_path}/tags/TagPost_{tag_id}"
            post_prim = create_prim(post_path, "Cylinder")

            post_geom = UsdGeom.Cylinder(post_prim)
            post_geom.CreateRadiusAttr(0.02)
            post_geom.CreateHeightAttr(post_height)

            post_xform = UsdGeom.Xformable(post_prim)
            post_xform_ops = post_xform.GetOrderedXformOps()
            if len(post_xform_ops) >= 1:  # translate
                post_xform_ops[0].Set(Gf.Vec3d(x, y, post_height / 2))
            else:
                post_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, post_height / 2))

            # Create tag plane
            tag_path = f"{self.stage_path}/tags/ArUco_{tag_id}"
            tag_prim = create_prim(tag_path, "Cube")

            tag_geom = UsdGeom.Cube(tag_prim)
            tag_geom.CreateSizeAttr(1.0)

            tag_xform = UsdGeom.Xformable(tag_prim)
            tag_xform_ops = tag_xform.GetOrderedXformOps()
            if len(tag_xform_ops) >= 3:  # translate, orient, scale
                tag_xform_ops[2].Set(Gf.Vec3d(tag_size, tag_size, 0.001))  # scale
                tag_xform_ops[0].Set(Gf.Vec3d(x, y, z))  # translate
            else:
                tag_xform.AddScaleOp().Set(Gf.Vec3d(tag_size, tag_size, 0.001))
                tag_xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

            # Store tag info for YAML export
            tag_map[tag_id] = {
                "position": [float(x), float(y), float(z)],
                "orientation": [0.0, 0.0, 0.0, 1.0],  # quaternion
                "size": float(tag_size),
            }

        return tag_map

    def generate_world(self, output_path):
        """Generate complete world and save USD"""
        # Create arena components
        self.create_arena()
        self.create_central_obstacle()
        tag_map = self.create_aruco_tags()

        # Save USD file
        self.stage.GetRootLayer().Export(output_path.as_posix())

        # Save tag map YAML
        yaml_path = output_path.as_posix().replace(".usd", "_aruco_map.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"tags": tag_map}, f, default_flow_style=False)

        print(f"Generated world: {output_path}")
        print(f"Generated tag map: {yaml_path}")


# This file is now deprecated - arena creation has been moved to go1_nav_env.py
# Keep this file for reference or delete if not needed

# The arena creation functions have been moved to go1_nav_env.py
# and now use Isaac Lab's sim_utils instead of direct USD manipulation
