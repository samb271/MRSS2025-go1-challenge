#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    # Launch arguments
    headless_arg = DeclareLaunchArgument(
        "headless", default_value="false", description="Run Isaac Sim in headless mode"
    )

    world_file_arg = DeclareLaunchArgument(
        "world_file", default_value="isaac_sim/worlds/go1_nav_world.usd", description="Path to world USD file"
    )

    # Isaac Sim launch
    isaac_sim_cmd = ExecuteProcess(
        cmd=[
            "python3",
            "isaac_sim/scripts/go1_nav_env.py",
            "--world",
            LaunchConfiguration("world_file"),
            "--headless",
            LaunchConfiguration("headless"),
        ],
        output="screen",
    )

    # ArUco localization node
    aruco_node = Node(
        package="go1_challenge",
        executable="aruco_localization_node.py",
        name="aruco_localization",
        parameters=[{"tag_map_file": "isaac_sim/config/aruco_map.yaml", "camera_topic": "/camera/image_raw"}],
        output="screen",
    )

    # Baseline planner node
    planner_node = Node(
        package="go1_challenge",
        executable="baseline_planner.py",
        name="baseline_planner",
        parameters=[{"goal_x": 2.0, "goal_y": 2.0, "grid_resolution": 0.05, "arena_size": 5.0}],
        output="screen",
    )

    # Optional: Student policy node (comment out when using baseline)
    # student_node = Node(
    #     package='go1_challenge',
    #     executable='my_policy.py',
    #     name='student_policy',
    #     output='screen'
    # )

    return LaunchDescription(
        [
            headless_arg,
            world_file_arg,
            isaac_sim_cmd,
            aruco_node,
            planner_node,
            # student_node,
        ]
    )
