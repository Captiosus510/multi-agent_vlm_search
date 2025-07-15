import os
import pathlib
import launch
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.substitutions.path_join_substitution import PathJoinSubstitution
from launch_ros.actions import Node
from webots_ros2_driver.urdf_spawner import URDFSpawner, get_webots_driver_node
from webots_ros2_driver.webots_launcher import WebotsLauncher, Ros2SupervisorLauncher
from webots_ros2_driver.webots_controller import WebotsController



def generate_launch_description():
    package_dir = get_package_share_directory('llm_search')
    robot_description_path = os.path.join(package_dir, 'resource', 'my_robot.urdf')
    # other_robot_description_path = os.path.join(package_dir, 'resource', 'other_robot.urdf')

    webots = WebotsLauncher(
        world=os.path.join(package_dir, 'worlds', 'break_room.wbt')
    )

    my_robot_driver = WebotsController(
        robot_name='my_robot',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

  
    my_robot_mapper = Node(
        package='llm_search',
        executable='local_mapper',
        output='screen',
        parameters=[
            {'robot_name': 'my_robot'},
            {'show_maps': False},  # Enable map display for the main robot
        ]
    )

    
    other_robot_driver = WebotsController(
        robot_name='other_robot',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    other_robot_mapper = Node(
        package='llm_search',
        executable='local_mapper',
        output='screen',
        parameters=[
            {'robot_name': 'other_robot'},
            {'show_maps': False},  # Disable map display for the other robot
        ]
    )

    controller_node = Node(
        package='llm_search',
        executable='tb4_controller',
        output='screen'
    )

    # Camera viewer node
    camera_viewer_my_robot = Node(
        package='llm_search',
        executable='camera_display',
        output='screen',
        parameters=[
            {'robot_name': 'my_robot'}
        ]
    )
    camera_viewer_other_robot = Node(
        package='llm_search',
        executable='camera_display',
        output='screen',
        parameters=[
            {'robot_name': 'other_robot'}
        ]
    )

    global_mapper = Node(
        package='llm_search',
        executable='global_map_merger',
        output='screen',
        parameters=[
            {'robot_names': ['my_robot', 'other_robot']},
            {'show_maps': True}  # Enable global map display
        ]
    )

    return LaunchDescription([
        webots,
        my_robot_driver, 
        my_robot_mapper,
        other_robot_driver,
        other_robot_mapper,
        # camera_viewer_my_robot,
        # camera_viewer_other_robot,
        global_mapper,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit( # type: ignore
                target_action=camera_viewer_my_robot,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit( # type: ignore
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])