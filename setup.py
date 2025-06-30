from setuptools import setup

package_name = 'llm_search'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/my_world.wbt']))
data_files.append(('share/' + package_name + '/resource', ['resource/my_robot.urdf']))
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

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user.name@mail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_robot_driver = llm_search.my_robot_driver:main',
            'tb4_controller = llm_search.tb4_controller:main',
            'camera_display = llm_search.camera_display:main',
        ],
    },
)