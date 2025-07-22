# LLM Multi-Agent Search

This repo is contiains code for multi-agent LLM search (not complete yet). 

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

This should launch the simulator and the global camera.  

Open another terminal and stay in your workspace directory. Here you should run:

```
# must source setup.bash every new terminal instance
source install/setup.bash 
ros2 run llm_search vlm_services
```

Now you can speak with the VLM about what you are looking for. 
