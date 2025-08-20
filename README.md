# Multi-Agent VLM Search

This repo is contiains code for a system that interfaces with the user through natural language to carry out search and monitoring tasks with a multi-robot system. It uses GPT 4.1 to analyze the scene and extract semantic information. Then, using some image segmentation and mesh building, it makes use of the resulting grid to direct robots to grid cells. In the future, I hope to expand this work to fully carry out a search pipeline. For now, it simply directs robots to relevant locations based on the query and posts them as sentries to monitor for objects.

![alt text](https://github.com/Captiosus510/,ulti-agent_vlm_search/blob/main/res.png?raw=true)

## Installation

Start by setting up a workspace
```
mkdir ~/your_workspace
cd ~/your_workspace
mkdir src
cd src
```

Then clone the repository. After cloning the repository:

```
cd ~/your_workspace
python -m venv venv_name
source venv/bin/activate
pip install -r src/llm_search/requirements.txt
```

## Running the code

Go back to your workspace and then run:
```
colcon build --symlink-install
source install/setup.bash
```

Then run:
```
ros2 launch llm_search world_launch.py
```

This should launch the simulator, robots, and connect the robot driving code to each. 

Open another terminal and stay in your workspace directory. Activate your venv as well. Here you should run:

```
# must source setup.bash every new terminal instance and make sure venv is activated
source install/setup.bash 
ros2 run llm_search vlm_services --ros-args -p num_robots:=(integer of number of robots in the simulation)
```

Now you can speak with the VLM about what you are looking for. 

