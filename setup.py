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
    'protos/pib_meshes/body_visual.dae',
    'protos/pib_meshes/bumper_visual.dae',
    'protos/pib_meshes/camera_bracket.dae',
    'protos/pib_meshes/rplidar.dae',
    'protos/pib_meshes/tower_sensor_plate.dae',
    'protos/pib_meshes/tower_standoff.dae',
    'protos/pib_meshes/weight_block.dae'
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