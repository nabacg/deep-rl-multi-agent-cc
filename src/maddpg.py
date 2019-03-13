import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import CriticQNetwork, ActorQNetwork
from replaybuffers import ReplayBuffer
from utils import OUNoise


def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        

class MADDPGAgent:

    def __init__(self, 
                 agent_index,
                 state_size, 
                 action_size, 
                 seed, 
                 actor_lr, 
                 critic_lr, 
                 weight_decay,
                 tau,
                 update_every, gamma, device, 
                 hidden_1_size = 256,
                 hidden_2_size = 128,
                 checkpoint_dir="."):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): how many agents are running in each step
            seed (int): random seed
            actor_lr (float): actor learning rate alpha
            critic_lr (float): critic learning rate alpha
            weight_decay (float): rate of nn weight decay for critic network
            tau (float): soft update rate for synchronizing target and train network weights 
            update_every (int): how many env steps to train agent
            gamma (float): reward discount factor
            device (string): device to run PyTorch computation on (CPU, GPU)
            checkpoint_dir (string) : where to save checkpoints (trained weights)
        """
        self.agent_index = agent_index
        self.seed = torch.manual_seed(seed)
        self.tau = tau
        self.action_size = action_size
        
        
        self.update_every = update_every
        self.gamma = gamma 
        self.device = device
       
        
        # NN models for 
#         network size 
        
        
        # Critic 
        self.critic_train     = CriticQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.critic_target    = CriticQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_train.parameters(), lr=critic_lr, weight_decay=weight_decay)
        # Actor
        self.actor_train      = ActorQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.actor_target     = ActorQNetwork(state_size, action_size, seed, hidden_1_size, hidden_2_size).to(device)
        self.actor_optimizer  = optim.Adam(self.actor_train.parameters(), lr=actor_lr)
        
        # init Noise process
        self.noise = OUNoise(action_size, seed, theta=0.15, sigma=0.2)

        self.actor_loss = 0
        self.critic_loss = 0
        
        #checkpointing
        self.checkpoint_dir   = checkpoint_dir
        
        self.actor_weights =  "{}/actor_{}.pth".format(self.checkpoint_dir, self.agent_index)
        self.critic_weights = "{}/critic_{}.pth".format(self.checkpoint_dir, self.agent_index)
        
        
    def load_checkpoint(self, file_prefix=None):
        actor_weights =  "{}actor_{}.pth".format(file_prefix, self.agent_index)  if file_prefix else self.actor_weights
        critic_weights = "{}critic_{}.pth".format(file_prefix, self.agent_index)  if file_prefix else self.critic_weights

        if os.path.isfile(actor_weights) and os.path.isfile(critic_weights):
            self.actor_target.load_state_dict(torch.load(actor_weights))
            self.actor_train.load_state_dict(torch.load(actor_weights))
            self.critic_target.load_state_dict(torch.load(critic_weights))
            self.critic_train.load_state_dict(torch.load(critic_weights))
            
    def save_checkpoint(self, file_name=None):
        actor_weights =  "{}actor_{}.pth".format(file_prefix, self.agent_index)  if file_name else self.actor_weights
        critic_weights = "{}critic_{}.pth".format(file_prefix, self.agent_index) if file_name else self.critic_weights
        torch.save(self.actor_train.state_dict(), actor_weights)      
        torch.save(self.critic_train.state_dict(), critic_weights)  

    def act(self, states, add_noise = True, epsilon=1.0):
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_train.eval()
        with torch.no_grad():
            actions = self.actor_train(states).cpu().data.numpy()
        self.actor_train.train()
        
        if add_noise:
            actions += self.noise.sample()*epsilon
            # actions += 0.5*np.random.standard_normal(self.action_size) *epsilon
        return np.clip(actions, -1, 1)
    

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, target_actions_pred, actions_pred = experiences
        
       
        ## DDPG  implementation 
        #### Critic network training

        # Calculate Q_Targets
        # first use target Actor to predict best next actions for next states S'
        # target_actions_pred = self.actor_target(next_states)
        # Then use target critic to asses Q value of this (S', pred_action) tuple
        with torch.no_grad():
            target_pred = self.critic_target(next_states, target_actions_pred).to(self.device)
        # calculate the Q_target using TD error formula   
        Q_target = rewards[:, self.agent_index].view(-1, 1)  + (self.gamma * target_pred * (1 - dones[:, self.agent_index].view(-1, 1) ))
        
        # find what Q value does Critic train network assign to this (state, action) - current state, actual action performed        
        Q_pred = self.critic_train(states, actions).to(self.device)
        
        # Minimize critic loss
        # do Gradient Descent step on Critic train network by minimizing diff between (Q_pred, Q_target)
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(Q_pred, Q_target.detach())
        self.critic_loss = critic_loss.cpu().detach().item()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        #### Actor network training
        # find wich action does Actor train predict
        # actions_pred = self.actor_train(states)
        # Loss is negative of Critic_train Q estimate of (S,  actions_pred)
        # i.e. we want to maximize (minimize the negative) of action state Value function (Q) prediction by critic_train 
        # for current state and next action predicted by actor_train
        actor_loss = -self.critic_train(states, actions_pred).mean()
        
        self.actor_loss = actor_loss.cpu().detach().item()
        # minimize Actor loss
        # do Gradient Descent step on Actor train network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ------------------- update target network ------------------- #
        soft_update(self.critic_train, self.critic_target, self.tau)
        soft_update(self.actor_train, self.actor_target, self.tau)