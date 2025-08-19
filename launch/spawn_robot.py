#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: spawn_robot.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    Launch file for spawning the robot in the simulation environment. 
    It does a variety of tasks to setup the robot for use in other scripts:
    - intialize robot controller with webots, publishing sensor data to ROS2 topics
    - loads YOLO based object detector for goal detection
    - loads camera display node for visualizing camera feed
    - loads the navigator node for robot navigation and motion planning
"""

import os
import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController


from launch.actions import OpaqueFunction

def launch_setup(context, *args, **kwargs):
    package_dir = get_package_share_directory('llm_search')
    robot_description_path = os.path.join(package_dir, 'resource', 'tb4.urdf')
    robot_name = LaunchConfiguration('robot_name').perform(context)
    robot_speed = LaunchConfiguration('robot_speed').perform(context)
    robot_turn_speed = LaunchConfiguration('robot_turn_speed').perform(context)
    behavior = LaunchConfiguration('behavior').perform(context)

    # Read URDF content
    with open(robot_description_path, 'r') as urdf_file:
        robot_description_content = urdf_file.read()

    robot_controller = WebotsController(
        robot_name=robot_name,
        parameters=[
            {'robot_description': robot_description_content},
            {'use_sim_time': True},
            {'publish_tf': True},
            {'publish_odom': True},
            {'tf_prefix': robot_name},
        ]
    )

    detector = Node(
        package='llm_search',
        executable='detector',
        name='object_detector',
        output='screen',
        parameters=[
            {'robot_name': robot_name}
        ]
    )

    camera = Node(
        package='llm_search',
        executable='camera_display',
        name='camera_viewer',
        output='screen',
        parameters=[
            {'robot_name': robot_name},
            {'camera_topic': '/detector/image'},
            {'show_depth': False},  # Assuming depth camera is available
            {'show_rgb': False}
        ]
    )

    navigator = Node(
        package='llm_search',
        executable='navigator',
        name='navigator',
        output='screen',
        parameters=[
            {'robot_name': robot_name},
            {'robot_speed': float(robot_speed)},
            {'robot_turn_speed': float(robot_turn_speed)},
            {'behavior': behavior}  # or 'search'
        ]
    )

    return [
        robot_controller,
        detector,
        camera,
        navigator,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit( # type: ignore
                target_action=robot_controller,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ]

def generate_launch_description():
    robot_name_arg = DeclareLaunchArgument(
        'robot_name',
        default_value='my_robot',
        description='Name of the robot to spawn'
    )
    return LaunchDescription([
        robot_name_arg,
        OpaqueFunction(function=launch_setup)
    ])