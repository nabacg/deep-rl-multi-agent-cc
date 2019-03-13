# deep-rl-multi-agent-cc
This repository contains code demonstrating how to implement Deep Reinforcement Learning techniques to solve a Unity ML-Agents environment where a goal is to train an agent to solve the variation of [Tennis environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis), where the 2 agents control their rackets separately and their goal is to keep the ball from touchnig the floor for as long as possible. It's a great example of multi agent environment where agents are required to cooperate in order to solve the problem. Its also an example of continuous control, where Action space consists of 2 floating point numbers.

Solution involves implementation of Multi Agent Deep Deterministic Policy Gradient or MADDPG as described in [OpenAI publication](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) with 2 agents to solve a cooperative task. MADDPG is as an example of actor-critic, model-free reinforcment learning algorithms suitable for problems with continuous action space, in  cooperative, competitive or mixed scenarios.

Implementation using Python 3, PyTorch, Unity ML-Agents.

## Demo
<img src="https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/tennis_env.gif?raw=true" width="360" height="240" border="10" />

[Longer demo on youtube](https://youtu.be/5gzChrLv5pw)

## Tennis environment 

- Set-up: Two-player game where agents control rackets to bounce ball over a net.
- Goal: The agents must bounce ball between one another while not dropping or sending ball out of bounds.
- Agents: The environment contains two agent linked to a single Brain named TennisBrain.
- Agent Reward Function (independent):
 - +0.1 To agent when hitting ball over net.
 - -0.1 To agent who let ball hit their ground, or hit ball out of bounds.
- Observation space: Vector Observation space: 8 variables corresponding to position and velocity of ball and racket.
- Action space: (Continuous) Size of 2, corresponding to movement toward net or away from net, and jumping.
- Benchmark Mean Reward: 0.5

## Solution 
Multiple cooperating Deep Reinforcement Learning agents implementation presented here solves this environment in a little over 819 episodes, details presented in
- [Report](Report.md) - for discussion on results
- [Training](Training.ipynb) - for details on training and how to use this code

## Project structure

- src - python source files
    - maddpg.py contains Agent and Critic implementation and some helper functions
    - model.py - contains PyTorch Neural Network modules for for Actor and Critic
    - replaybuffers.py - contain Experience Replay and Prioritized Experience Replay helper classes
    - utils.py - contains OUNoise class, plus some helper functions useful for training and testing
    - main.py - delivers easy command line interface into other classes
- model_weights - folder contains actor and critic pre-trained weights



# Instructions 

# Getting Started
Before running the code you'll need to install following python modules

## Requirements 

- [Python 3](https://www.python.org/) - currently only python 3.6 is supported due to Unity ML-Agents requirements
- [PyTorch](https://www.pytorch.org)
- [Numpy](http://www.numpy.org/)
- [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) 
- [Jupyter Notebooks](https://jupyter.org/) - optional, needed to run [Report notebook](Report.ipynb) and [Training notebook](Training.ipynb) notebooks

## Installing requirements with Anaconda
Easiest way to install all dependencies is using [Anaconda](https://www.anaconda.com/distribution/). Install Anaconda for Python 3 using installer appropriate for your OS and once ready clone this repository and environment.yml file inside it.

```bash
git clone https://github.com/nabacg/deep-rl-multi-agent-cc.git
cd deep-rl-multi-agent-cc
conda env create -f environment.yml

```
It will take few minutes to install all packages. Once finished activate the newly created environment with

```bash
conda activate drl_nav
``` 

## Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:


- Linux: ["click here"](href="https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: ["click here"](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the root folder in the deep-rl-multi-agent-cc GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)



## Jupyter notebooks
In order to train DQN Agent using Jupyter notebooks provided, start jupyter in project folder:

```bash
cd deep-rl-multiagent
jupyter notebook 
``` 

then once Jupyter UI opens in your browser ([the default URL](http://localhost:8888/tree/) ),  open [Training notebook](Training.ipynb). 





## Command Line

It's also possible to train or test DQN Agent using command line only with help of main.py file. 

For example to train agent from scratch for 2000 episodes or until mean score of 13 is reached use this command:

```bash
python src/main.py --episodes 2000 --target_score 0.5
```

 To test already pretrained agent for 100 episodes using pretrained model from  qnetwork_model_weights.pth use: 
```bash

 python src/main.py --episodes 100 --mode test --input_weights_prefix solution_model

```

The file exposes several command line arguments that allow to change various (hyper)parameters, those can be displayed using --help argument.

```bash
python src/main.py --help
usage: main.py [-h] [--env_file ENV_FILE] [--mode {train,test}]
               [--episodes EPISODES] [--target_score TARGET_SCORE]
               [--input_weights_prefix INPUT_WEIGHTS_PREFIX]
               [--output_weights_prefix OUTPUT_WEIGHTS_PREFIX]

optional arguments:
  -h, --help            show this help message and exit
  --env_file ENV_FILE   Path to Unity Environment file, allows to change which
                        env is created. Defaults to Banana.app
  --mode {train,test}   Allows switch between training new DQN Agent or test
                        pretrained Agent by loading model weights from file
  --episodes EPISODES   Select how many episodes should training run for.
                        Should be multiple of 100 or mean target score
                        calculation will not make much sense
  --target_score TARGET_SCORE
                        Target traning score, when mean score over 100
                        episodes
  --input_weights_prefix INPUT_WEIGHTS_PREFIX
                        Path prefix that will be appended with
                        _{actor|critic}.pth load model weights
  --output_weights_prefix OUTPUT_WEIGHTS_PREFIX
                        Path prefix that will be appended with
                        _{actor|critic}.pth save Q Networks model weights
                        after training.

```
