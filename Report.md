# Report

# Results
## Plot of Rewards

Below graphs show solutions to Tennis environment, scores   Plot from [Training](Training.ipynb) showing scores (blue line, max of total rewards for 2 agents for episode) and a mean of those scores over 100 episode window. The graph shows agent until it achieves a target mean score of 0.61, which happend after 819 episodes. Below are graphs of loss functions for 2 actor networks and 2 critic networks trained.

### Rewards per episode 
Graph shows max of both agents accumulated rewards per episode (blue line) and mean of this score over 100 episode window (red line)
![Plot of agent scores by episode](https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/061_scores.png?raw=true)


### Actors Loss function
![Plot of Actor network loss function](https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/061_actor_losses.png?raw=true)

### Critics Loss function
![Plot of Critic network loss function](https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/061_critic_losses.png?raw=true)


## Learning Algorithm 

This repository, specifically files below:
 - src/
     - maddpg.py
     - utils.py
     - model.py 
     - replaybuffers.py


Contain implementation of Multi Agent Deep Deterministic Policy Gradient or MADDPG as described in [OpenAI publication](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) where 2 agents instances of (MADDPGAgent defined in src/maddpg.py) independently train separate policy network (Actor) and each contain a separatly trained copy of centralized Critic network. 

Best shown on this diagram from orignal publication:
![MADDPG diagram](https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/MADDPG_diagram.png?raw=true)

MADDPG Algorithm pseudo code 
![MADDPG algorithm](https://github.com/nabacg/deep-rl-multi-agent-cc/blob/master/images/MADDPG_algorithm.png?raw=true)


## Exploration vs Exploitation
I found that forcing agents to explore a lot in first stage of training was necessary to solve the environment in reasonable time, otherwise agents tended to barely take any actions, often staying motionless next to the net. In order to achive that I introduced to take fully random actions for first 20% of training episodes, just by drawing action vectors from standard normal distribution. After that period I switch back to Ornstein-Uhlenbeck process with theta = 0.15 and sigma = 0.2, but I scale noise volume drawn from it by epsilon value. Epsilon starts at value of 0.9 and decays by 0.999 every 10 episodes down to the minimum of 0.01. Below code snippet from utils.py train_agent function shows the importants explained above:

```python
 eps = 0.9
 #...
 if i_episode < 0.2 * n_episodes:
        actions = np.random.standard_normal((num_agents, action_size))
    else:
        actions   =  np.vstack( [a.act(s, add_noise=True, epsilon=eps) for (a,s) in zip(agents, states[:]) ])  

#...
if step_counter % 10 == 0:
    eps = max(eps_end, eps_decay*eps)         
```

Introduction of this greatly improved the learning speed.

### Neural Network architecture 

File src/model.py contains PyTorch implementation of two small Neural Networks, for approximating the action value function Q (the Critic) and the Policy function (Actor). 


#### Actor network 
Consists of 4 fully connected layers of following size:
 - Input layer (state_size, 256), ReLU activation 
 - Hidden layer (256, 256), ReLU activation
 - Another hidden layer (256, 128), ReLU activation
 - Output layer (128, action_size), Tanh activation

Below output of how Pytorch prints out Actor network architecture
```python 
ActorQNetwork(
  (fc_1): Linear(in_features=24, out_features=256, bias=True)
  (fc_2): Linear(in_features=256, out_features=256, bias=True)
  (fc_3): Linear(in_features=256, out_features=128, bias=True)
  (output): Linear(in_features=128, out_features=2, bias=True)
)

```
 #### Critic network
Consists of 4 fully connected layers of following size:
 - Input layer (52, 512), ReLU activation, input size of 52 comes from (state_size+action_size)*2
 - Hidden layer (256, 256), ReLU activation
 - Hidden layer (256, 128), ReLU activation
 - Output layer (256, 1), Linear activation

Below output of how Pytorch prints out Critic network architecture
```python
CriticQNetwork(
  (fc_1): Linear(in_features=52, out_features=256, bias=True)
  (fc_2): Linear(in_features=256, out_features=128, bias=True)
  (fc_3): Linear(in_features=128, out_features=128, bias=True)
  (output): Linear(in_features=128, out_features=1, bias=True)
)
```

For Tennis environment 

```python
state_size = 24
action_size = 2
```

The size and number of hidden layers was chosen experimentally.

### NN Training
Critic network was trained by minimizing [Smooth L1 Loss function](https://pytorch.org/docs/stable/nn.html#smooth-l1-loss) with help of [ADAM optimizer](https://pytorch.org/docs/stable/optim.html?highlight=mseloss#torch.optim.Adam). I've found L1 loss function performed better then Mean Squared error in my experiments.

Learning rate of  LR_ACTOR = 1e-4 for Actor network and LR_CRITIC = 1e-3 for Critic network, was chosen after some experimentation with values between (1e-2, 1e-5). All chosen hyperparameter values are listed below. [Training notebook](Training.ipynb) contains an executable demonstration of training process and results.  


### Model weights
Pretrained weights used to generate results presented here are part of this repository
and can be found in folder model_weights:
- 2 Actor model parameters:
 - 061_solution_0_actor.pth  
 - 061_solution_1_actor.pth
- 2 Critic model parameters:
 - 061_solution_0_critic.pth  
 - 061_solution_1_critic.pth

## Hyperparameters used

```python

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters               
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic

UPDATE_EVERY = 1        # how often to update the network
WEIGHT_DECAY = 0
hidden_1_size = 256
hidden_2_size = 128
```




#  Ideas for Future Work
- test this solution on more environments with continuous control, especially with larger number of agents and mixuture of collaboration and competition
- I've found this model to be slow and rather unstable to train so it would worthwile implementing some technique to train it more efficiently. Perhaps some of the original DQN ideas would be a good start
 - Prioretized Experience Replay
 - Double or Dueling DQN or even Rainbow algorithm
- try implementing Policy Ensembles which should improve performance as suggested in MADDPG paper
