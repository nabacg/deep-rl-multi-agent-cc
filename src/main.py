import argparse
import os
import sys
import numpy as np
import torch 
from ddpg_agent import DdpgAgent
from utils import train_agent, test_agent, plot_scores_losses
from unityagents import UnityEnvironment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', default="Reacher.app", help="Path to Unity Environment file, allows to change which env is created. Defaults to Banana.app")
    parser.add_argument('--mode', choices=["train", "test"], default="train", help="Allows switch between training new DQN Agent or test pretrained Agent by loading model weights from file")
    parser.add_argument('--episodes', type=int, default=2000, help="Select how many episodes should training run for. Should be multiple of 100 or mean target score calculation won't make much sense")
    parser.add_argument('--target_score', type=float, default=30.0, help="Target traning score, when mean score over 100 episodes ")
    parser.add_argument('--input_weights_prefix', type=str, default=None, help="Path prefix that will be appended with _{actor|critic}.pth load model weights")
    parser.add_argument('--output_weights_prefix', type=str, default="new_model", help="Path prefix that will be appended with _{actor|critic}.pth save Q Networks model weights after training.")
    

    args = parser.parse_args()

    train_mode = args.mode == "train"
    

    env = UnityEnvironment(file_name=args.env_file, no_graphics=not(train_mode))
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    print("Resetting env to {} mode".format(args.mode))
    env_info = env.reset(train_mode=train_mode)[brain_name]
    num_agents = len(env_info.agents)
    # # number of agents in the environment
    # print('Number of agents:', len(env_info.agents))

    # # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # # examine the state space
    state = env_info.vector_observations[0]
    # print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 128         # minibatch size
    GAMMA = 0.99            # discount factor
    TAU = 1e-3              # for soft update of target parameters               
    LR_ACTOR = 1e-4         # learning rate of the actor 
    LR_CRITIC = 1e-4        # learning rate of the critic

    UPDATE_EVERY = 1        # how often to update the network
    WEIGHT_DECAY = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    agent = DdpgAgent(state_size  = state_size, 
                    action_size = action_size, 
                    num_agents = num_agents,
                    seed=2,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER_SIZE,
                    actor_lr=LR_ACTOR,
                    critic_lr=LR_CRITIC,
                    weight_decay=WEIGHT_DECAY,
                    tau=TAU,
                    update_every=UPDATE_EVERY,
                    gamma = GAMMA,
                    device = device,
                    checkpoint_dir = "checkpoints")

    if train_mode:
        input_weights = args.input_weights_prefix 
        train_agent(agent, env, args.output_weights, args.target_score, args.episodes, args.eps_decay, args.eps_end, input_weights)
    elif args.input_weights_prefix :
        test_agent(agent, env, args.input_weights_prefix, args.episodes)
    else:
        print("Test mode requires providing input_weights path to existing file! ")
