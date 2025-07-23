from setuptools import setup, find_packages

package_name = 'llm_search'
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
data_files.append(('share/' + package_name + '/protos/meshes', [
    'protos/meshes/body_visual.dae',
    'protos/meshes/bumper_visual.dae',
    'protos/meshes/camera_bracket.dae',
    'protos/meshes/rplidar.dae',
    'protos/meshes/tower_sensor_plate.dae',
    'protos/meshes/tower_standoff.dae',
    'protos/meshes/weight_block.dae'
]))

data_files.append(('share/' + package_name, ['package.xml']))
data_files.append(('share/' + package_name, ['slam_params.yaml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mahd Afzal',
    maintainer_email='afzalmahd@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_driver = llm_search.robot_driver:main',
            'navigator = llm_search.navigator:main',
            'camera_display = llm_search.camera_display:main',
            'vlm_services = llm_search.vlm_services:main',
            'local_mapper = llm_search.local_mapper:main',
            'global_map_merger = llm_search.global_mapper:main',
            'detector = llm_search.detector:main',
            'global_cams = llm_search.global_cams:main',
        ],
    },
)