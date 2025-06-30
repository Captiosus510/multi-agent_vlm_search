# LLM Multi-Agent Search

This repo is contiains code for multi-agent LLM search (not complete yet). 

## Installation

First clone the repository and cd into it then:

```
pip install -r requirements.txt
```

## Running the code

Go to your workspace and then run:
```
colcon build
source install/setup.bash
```

This needs to be run everytime you change the code.

Then run:
```
ros2 launch llm_search robot_launch.py
```
