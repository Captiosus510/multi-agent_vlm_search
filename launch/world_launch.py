#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: world_launch.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    Launch file for intializing the world and robots to spawn with.
    If you want more robots, you must also spawn them in with the <extern> controller and translation in the world wbt file. (look in the worlds directory)
"""

import os
import pathlib
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch_ros.actions import Node
from webots_ros2_driver.urdf_spawner import URDFSpawner, get_webots_driver_node
from webots_ros2_driver.webots_launcher import WebotsLauncher, Ros2SupervisorLauncher
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    package_dir = get_package_share_directory('multi_robot_allocation')
    robot_description_path = os.path.join(package_dir, 'resource', 'my_robot.urdf')
    # other_robot_description_path = os.path.join(package_dir, 'resource', 'other_robot.urdf')

    global_cam_path = os.path.join(package_dir, 'resource', 'global_cam.urdf')

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'break_room.wbt'),
        ros2_supervisor=True
    )

    global_cam = WebotsController(
        robot_name='global_cam',
        parameters=[
            {'robot_description': global_cam_path},
        ]
    )

    camera_viewer_global = Node(
        package='multi_robot_allocation',
        executable='camera_display',
        output='screen',
        parameters=[
            {'robot_name': 'global_cam'},
            {'show_depth': False},  # Global camera does not have depth
            {'show_rgb': False},
        ]
    )

    all_robo_cams = Node(
        package='multi_robot_allocation',
        executable='global_cams',
        output='screen',
        parameters=[
            {'topic_name': '/detector/image'},
        ]
    )

    robot_names = ['tb1', 'tb2']
    # include robot launch files
    spawn_robots = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(package_dir, 'launch', 'spawn_robot.py')),
            launch_arguments={
                'robot_name': name,
                'robot_speed': '0.25',
                'robot_turn_speed': '0.5',
                'behavior': 'monitor'
            }.items()
        ) for name in robot_names
    ]
    return LaunchDescription([
        webots,
        webots._supervisor,
        global_cam,
        all_robo_cams,
        camera_viewer_global]
        + spawn_robots
        + [
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit( # type: ignore
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])