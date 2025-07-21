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
        package='llm_search',
        executable='camera_display',
        output='screen',
        parameters=[
            {'robot_name': 'global_cam'},
            {'has_depth': False}  # Global camera does not have depth
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
        webots._supervisor,
        global_cam,
        camera_viewer_global,
        # global_mapper,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit( # type: ignore
                target_action=webots,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])