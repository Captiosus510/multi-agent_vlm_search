# Multi-Robot Allocation with VLM

This repo is contiains code for a system that interfaces with the user through natural language to carry out monitoring tasks with a multi-robot system. It uses GPT 4.1 to analyze and understand the scene. Then, using some image segmentation and mesh building, it makes use of the resulting grid to direct robots to grid cells. In the future, I hope to expand this work to fully carry out a search pipeline. For now, it simply directs robots to relevant locations based on the query and posts them as sentries to monitor for objects. The goal is to simplify the world's representation to leverage the VLM's strength in image analysis and semantics while keeping it from relying too much on spatial awareness.

This project will be used for later downstream tasks that involve directing multiple robots (manipulation, search, etc.)

Here is a flowchart for the initalization and the world building of the system:

![alt text](https://github.com/Captiosus510/multi-agent_vlm_search/blob/main/res/flowchart.png?raw=true)

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
pip install -r src/multi_robot_allocation/requirements.txt
```

## Running the code

Go back to your workspace and then run:
```
colcon build --symlink-install
source install/setup.bash
```

Then run:
```
ros2 launch multi_robot_allocation world_launch.py
```

This should launch the simulator, robots, and connect the robot driving code to each. 

Open another terminal and stay in your workspace directory. Activate your venv as well. Here you should run:

```
# must source setup.bash every new terminal instance and make sure venv is activated
source install/setup.bash 
ros2 run multi_robot_allocation vlm_services --ros-args -p num_robots:=(integer of number of robots in the simulation)
```

Now you can speak with the VLM about what you are looking for. 

## Results

![alt text](https://github.com/Captiosus510/multi-agent_vlm_search/blob/main/res/find_package_prompt.png?raw=true)

Prompt: I am waiting for a package. Could you monitor the entrances for me?

![alt text](https://github.com/Captiosus510/multi-agent_vlm_search/blob/main/res/cat_hiding_prompt.png?raw=true)

Prompt: My cat likes hiding behind plants. Can you monitor for him. 

## Acknowledgments

This project leverages several state-of-the-art technologies and open-source libraries that made this multi-robot allocation system possible:

### AI and Machine Learning
- **OpenAI GPT-4** - For natural language processing, semantic scene understanding, and intelligent task planning in our Vision-Language Model (VLM) services
- **Segment Anything Model (SAM)** by Meta AI - For advanced image segmentation and object detection capabilities
- **Ultralytics YOLO** - For real-time object detection and computer vision processing

### Computer Vision and Image Processing
- **OpenCV** - For comprehensive computer vision operations, image preprocessing, and visualization
- **cv_bridge** - For seamless conversion between ROS 2 image messages and OpenCV formats
- **NumPy** - For efficient numerical computations and array operations in image processing pipelines

### Robotics and Path Planning
- **ROS 2 (Robot Operating System)** - For the core robotics middleware, communication, and system architecture
- **Conflict-Based Search (CBS) Algorithm** - For multi-agent path finding and collision-free trajectory planning
- **TurtleBot4 Platform** - For the robotic hardware platform and simulation environment

### 3D Processing and Visualization
- **Matplotlib** - For data visualization, grid overlays, and real-time plotting of robot states
- **Trimesh** - For 3D mesh generation and geometric processing of the environment
- **Scipy** - For scientific computing and spatial data structures used in mesh triangulation

### Development and Utilities
- **Python 3** - The primary programming language for this project
- **Threading and Queue Libraries** - For concurrent processing and interrupt-driven task handling
- **Base64 Encoding** - For efficient image transmission to cloud-based AI services

### Special Thanks
We acknowledge the broader robotics and AI research community for advancing the fields of:
- Multi-agent systems and coordination
- Vision-language models for robotics

## Future Work

As stated previously, this project will be used for various downstream tasks. One such task is search, which has been partly implemented based on an idea presented in this paper:

Yokoyama, Naoki, et al. “VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation.” *arXiv preprint* arXiv:2312.03275, 6 Dec. 2023. 
[paper link](https://arxiv.org/pdf/2312.03275)

In the mapper branch, there is some code contained within local mapper that makes use of SigLip to generate a semantic map of the area around the robot. The idea is that the robot will go in the direction with the best confidence score in a search task. There is also framework in the global mapper file for collecting all the local maps from multiple robots into one global semantic map. 

Most of the initial part and direction of search is already set up. What needs to be implemented is the decentralized search phase that would come after the initial direction from the VLM.
