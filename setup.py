from setuptools import find_packages, setup

package_name = 'multi_robot_allocation'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/world_launch.py']))
data_files.append(('share/' + package_name + '/launch', ['launch/spawn_robot.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/my_world.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/break_room.wbt']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/detection_test.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/tb4.urdf']))
data_files.append(('share/' + package_name + '/resource', ['resource/global_cam.urdf']))
data_files.append(('share/' + package_name + '/protos', ['protos/Turtlebot4.proto']))
data_files.append(('share/' + package_name + '/protos/tb4_meshes', [
    'protos/tb4_meshes/body_visual.dae',
    'protos/tb4_meshes/bumper_visual.dae',
    'protos/tb4_meshes/camera_bracket.dae',
    'protos/tb4_meshes/rplidar.dae',
    'protos/tb4_meshes/tower_sensor_plate.dae',
    'protos/tb4_meshes/tower_standoff.dae',
    'protos/tb4_meshes/weight_block.dae'
]))

data_files.append(('share/' + package_name + '/protos', ['protos/pib.proto']))
data_files.append(('share/' + package_name + '/protos/pib_meshes', [
    'protos/pib_meshes/urdf_body.stl',
    'protos/pib_meshes/urdf_camera_link.stl',
    'protos/pib_meshes/urdf_elbow-lower.stl',
    'protos/pib_meshes/urdf_elbow-upper.stl',
    'protos/pib_meshes/urdf_finger_distal.stl',
    'protos/pib_meshes/urdf_finger_proximal.stl',
    'protos/pib_meshes/urdf_forearm.stl',
    'protos/pib_meshes/urdf_head.stl',
    'protos/pib_meshes/urdf_head_base.stl',
    'protos/pib_meshes/urdf_palm_left.stl',
    'protos/pib_meshes/urdf_palm_right.stl',
    'protos/pib_meshes/urdf_shoulder_horizontal.stl',
    'protos/pib_meshes/urdf_shoulder_vertical.stl',
    'protos/pib_meshes/urdf_thumb_rotator_left.stl',
    'protos/pib_meshes/urdf_thumb_rotator_right.stl'
]))

data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mahd Afzal',
    maintainer_email='afzalmahd@gmail.com',
    description='Multi-robot allocation package for ROS2',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'robot_driver = multi_robot_allocation.robot_driver:main',
            'navigator = multi_robot_allocation.navigator:main',
            'camera_display = multi_robot_allocation.camera_display:main',
            'vlm_services = multi_robot_allocation.vlm_services:main',
            'detector = multi_robot_allocation.detector:main',
            'global_cams = multi_robot_allocation.global_cams:main',
        ],
    },
)