# LLM Multi-Agent Search

This repo is contiains code for multi-agent LLM search (not complete yet). 

## Installation

Start by setting up a workspace
```
mkdir your_workspace
cd your_workspace
mkdir src
cd src
```

Then clone the repository. After cloning the repository:

```
cd llm_search
pip install -r requirements.txt
```

## Running the code

Go back to your workspace and then run:
```
colcon build
source install/setup.bash
```

This needs to be run everytime you change the code.

Then run:
```
ros2 launch llm_search robot_launch.py
```

This should launch the simulator and connect the controller. 

Open another terminal and stay in your workspace directory. Here you should run:

```
ros2 run llm_search vlm_services --ros-args -p input_prompt:="your prompt for the object you are searching for"
```

To control the robot manually, you may use the teleop keyboard functionality provided by ROS2 with a namespace:

```
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r /cmd_vel:=/my_robot/cmd_vel
```

You may control it in other ways too, the robot will take in twist messages published ont /my_robot/cmd_vel.
