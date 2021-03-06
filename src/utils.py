import random
import copy
import numpy as np
import torch
from collections import deque

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size        
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma        
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state        
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
    
def plot_scores_losses(scores, mean_scores, actors_losses, critic_losses):
    import matplotlib.pyplot as plt

    # len(agent.losses)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    plt.plot(np.arange(len(scores)), scores, label="Score")
    plt.plot(np.arange(len(mean_scores)), mean_scores, label="Mean Score", color="red")
    plt.ylabel('Scores')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()

    # ax11 = fig.add_subplot(2, 1, 2)
    # plt.plot(np.arange(len(mean_scores)), mean_scores)
    # plt.ylabel('Mean Score')
    # plt.xlabel('Episode #')
    # plt.show()

    actor1_losses = actors_losses[0]
    actor2_losses = actors_losses[1]
    ax2 = fig.add_subplot(2, 2, 3)
    
    plt.plot(np.arange(len(actor1_losses)), actor1_losses, label="Actor 1 loss", color="red")
    plt.plot(np.arange(len(actor2_losses)), actor2_losses, label="Actor 2 loss", color="blue")
    # plt.plot(np.arange(len(critic_losses)), critic_losses, label="Critic loss", color="green")
    plt.ylabel('Actor Losses')
    plt.xlabel('Steps #')
    plt.legend()
    
    plt.show()

    # actor2_losses = actors_losses[1]
    # ax3 = fig.add_subplot(3, 2, 3)
    # plt.plot(np.arange(len(actor2_losses)), actor2_losses)
    # plt.ylabel('Actor 2 Losses')
    # plt.xlabel('Steps #')
    # plt.show()

    ax4 = fig.add_subplot(2, 2, 4)
    # plt.plot(np.arange(len(critic_losses)), critic_losses)
    plt.plot(np.arange(len(critic_losses[0])), critic_losses[0], label="Critic 1 loss", color="red")
    plt.plot(np.arange(len(critic_losses[1])), critic_losses[1], label="Critic 2 loss", color="blue")
    plt.ylabel('Critic Losses')
    plt.xlabel('Steps #')
    plt.legend()
    
    plt.show()

def split_state(s, chunk_size):
    #todo make it flexible size, not hardcoded to 2 agents!
    # np.hsplit(ns, 2) might help
    return [s[:, :chunk_size], s[:, chunk_size:]]


def step(memory, agents, actor_losses, critic_losses, state_size):
    for (i,agent) in enumerate(agents):
        states, actions, rewards, next_states, dones = memory.sample()
        actor_target_predictions = torch.cat([a.actor_target(ns) 
                                    for (a,ns) in zip(agents, split_state(next_states, state_size))], dim=1)
        actor_train_predictions = torch.cat([ a.actor_train(s) if a.agent_index == agent.agent_index else a.actor_train(s).detach()
                                    for (a,s) in zip(agents, split_state(states, state_size))], dim=1)
        experiences = (states, actions, rewards, next_states, dones, actor_target_predictions, actor_train_predictions)
    
        agent.learn(experiences)
        actor_losses[i].append(agent.actor_loss)
        critic_losses[i].append(agent.critic_loss) 

def train_agent(agents, memory, env, action_size, batch_size, update_every,
         file_prefix, 
        print_metrics_every=10,
        target_mean_score=0.5, n_episodes = 1000, eps_decay=0.999, 
        eps_end=0.01,  score_aggregate = np.mean):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
   
    num_agents = len(agents)
    
    # memory = agents[0].memory
    scores = []
    mean_scores = []
    actor_losses = [[] for _ in agents]
    critic_losses = [[] for _ in agents]
    eps = 0.9
    max_score = 0
    # ToDo save weights for the model with best score so far
    step_counter = 0
    scores_window = deque(maxlen=100)  # last 100 scores
    for i_episode in range(1,n_episodes+1):
        # print("Episonde {} start".format(i_episode), end="")

        env_info = env.reset(train_mode=True)[brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current state
        state_size = states.shape[1]
        score = np.zeros(num_agents)                                           # initialize the score
        done = False                                          
        while not(done):                                 # exit loop if episode finished
            if i_episode < 0.2 * n_episodes:
                actions = np.random.standard_normal((num_agents, action_size))
            else:
                actions   =  np.vstack( [a.act(s, add_noise=True, epsilon=eps) for (a,s) in zip(agents, states[:]) ])                 # select an action
            # print(actions, actions.shape)
            env_info = env.step(actions)[brain_name]      # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                  # see if episode has finished
            # will populate for both agents

            # flip states, next_states and rewards so it can be fed into MADDPG Critic network with 2*state_size + 2*action_size
            memory.add(states.reshape(1, state_size*2), 
                        actions.reshape(1, action_size*2), 
                        rewards, 
                        next_states.reshape(1, state_size*2), 
                        dones)
            
            
            if len(memory) >= batch_size and step_counter % update_every == 0:
                # train   
                step(memory, agents, actor_losses, critic_losses, state_size)   
                
            score += rewards                              # update the score
            states = next_states                             # roll over the state to next time step
            done = np.any(dones)
            step_counter += 1
        
        score = score_aggregate(score)
        scores_window.append(score)
        scores.append(score)
        
        if step_counter % 10 == 0:
            eps = max(eps_end, eps_decay*eps)
        mean_score = np.mean(scores_window)
        mean_scores.append(mean_score)
        print('\rEpisode {}\tAverage Score: {:.2f}, epsilon={:.4f}'.format(i_episode, mean_score, eps), end="")
        [a.save_checkpoint() for a in agents]
      
        if i_episode % print_metrics_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
            plot_scores_losses(scores, mean_scores, actor_losses, critic_losses)
        if mean_score >= target_mean_score:
            print("Target mean score of {:.2f} achived at {:.2f} after {} episodes.".format(target_mean_score, mean_score, i_episode))
            [a.save_checkpoint(file_name="{}_{}".format(file_prefix, i)) for (i,a) in enumerate(agents)]
            agents[0].critic.save_checkpoint(file_name=file_prefix)
            break
    
    return scores, actor_losses, critic_losses

def test_agent(agents, env, weight_file_prefix, n_episodes):
    brain_name = env.brain_names[0]
    
    [a.load_checkpoint(weight_file_prefix) for a in agents]
    scores = [] 
    for i_episode in range(1,n_episodes+1):

        env_info = env.reset(train_mode=False)[brain_name]   # reset the environment
        states = env_info.vector_observations                # get the current state
        score = 0                                           # initialize the score
        done = False                                                                                   
        while not(done):                                 # exit loop if episode finished
            actions =  np.vstack( [a.act(s, add_noise=True) for (a,s) in zip(agents, states[:]) ])                 # select an action
            # print(actions, actions.shape)
            env_info = env.step(actions)[brain_name]      # send the action to the environment
            next_states = env_info.vector_observations   # get the next state
            rewards = env_info.rewards                   # get the reward
            dones = env_info.local_done                    # select an action
            # print(actions, actions.shape)
                

            score += np.mean(rewards)                           # update the score
            states = next_states                             # roll over the state to next time step
            done = np.any(dones)                          # roll over the state to next time step
        scores.append(score)
        print('\rEpisode {}\ Score: {:.2f}'.format(i_episode, score))
    return scores