import argparse
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from collections import deque
from ddpg_agent import Agent
from unityagents import UnityEnvironment


def train_agent(watch=False):
    """Train an agent for an environment.

    If argument `watch` is set to 'True', the user can watch the agent during
    training. Default is 'False'

    Parameters
    ----------
    watch : bool (Default False)
        Shows environment during training if set to 'True'.

    Returns
    -------
        A list of the scores of all episodes collected during training.
        If the environment was solved by the agent, its parameters
        (state dictionar) is saved.
    """
    config = Config.get_config()
    
    # Collection information regarding the environment and the agent (brain)
    env = UnityEnvironment(file_name=config.reacher_env,
                           no_graphics=not watch)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    state_size = brain.vector_observation_space_size
    # Initialize environment and get number of agents
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = env._n_agents[brain_name]
    
    # Initialize agent
    agent = Agent(state_size=state_size, 
                  action_size=action_size)
        
    # List containing scores from each episode
    scores_list = [] 
    # Queue containing the amount of scores specified by scores_window
    scores_window = deque(maxlen=config.scores_window)
    #eps = config.epsilon_start  # initialize epsilon
    #beta = config.beta_start    # initialize beta
    for i_episode in range(1, config.n_episodes+1):
        # Reset the environment and agent
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        # Obtain state of the initial environment
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        # Play the game until it terminates (done) or at most max timesteps
        for t in range(config.max_timesteps):
            # Get the actions for the current states from the agent
            actions = agent.act(states, add_noise=config.add_noise)
            # Take the next step using the received actions
            env_info = env.step(actions)[brain_name]
            # Extract the next states from the environment
            next_states = env_info.vector_observations
            # Get the rewards for the last actions
            rewards = env_info.rewards
            # Find out if the game is over (True) or still running (False)
            dones = env_info.local_done
            # Inform the agent about the results of the last step and
            # allow the agent to learn from it.
            for state, action, reward, next_state, done in \
                zip(states, actions, rewards, next_states, dones): 
                agent.step(state, action, reward, next_state, done)
            # Roll over the states to next time step and update score
            states = next_states
            scores += rewards
            # Exit loop if episode has finished
            if np.any(dones):
                break 

        # Save most recent score
        score = scores.mean()
        scores_window.append(score)
        scores_list.append(score)
        
        print_newline_after = 10 if num_agents > 1 else 100

        # Print result of the episode on the terminal screen
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % print_newline_after == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= config.total_avg_reward:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: \
                  {:.2f}'.format(i_episode, np.mean(scores_window)))
            # Save agent parameters
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
    env.close()
    return scores_list

def run_model(actor_params_file, critic_params_file):
    config = Config.get_config()
    env = UnityEnvironment(file_name=config.reacher_env)
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Number of states and actions
    state_size = brain.vector_observation_space_size
    action_size = brain.vector_action_space_size
    # Create default (untrained) agent
    agent = Agent(state_size=state_size, 
                  action_size=action_size)
    # Load trained parameters
    agent.actor_local.load_state_dict(torch.load(actor_params_file))
    agent.critic_local.load_state_dict(torch.load(critic_params_file))

    # Reset the environment
    env_info = env.reset(train_mode=False)[brain_name] # Reset the environment
    states = env_info.vector_observations              # Get the current states
    scores = np.zeros(env._n_agents[brain_name])       # Initialize the scores
    while True:
        actions = agent.act(states)                    # Select next actions
        env_info = env.step(actions)[brain_name]       # Send the actions to 
                                                       # environment
        next_states = env_info.vector_observations     # Get the next states
        rewards = env_info.rewards                     # Get the rewards
        dones = env_info.local_done                    # See if episodes have 
                                                       # finished
        scores += rewards                              # Update scores
        states = next_states                           # Roll over the states
                                                       # to next time step
        if np.any(dones):                              # Exit loop if any
           break                                       # episode finished
    
    # Print final scores
    print("Scores of each agent: {}".format(scores))
    print('Total score (averaged over agents): {}'.format(scores.mean()))
    # Close environment after three seconds
    time.sleep(3)
    env.close()


if __name__ == '__main__':
    # Process user arguments to start the requested function
    parser = argparse.ArgumentParser(
        description="Train or run an agent to navigate in a large square \
                     world collecting yellow bananas.")
    parser.add_argument('action', type=str,
                        help="Enter 'train' for agent training and 'run' for \
                              running a trained agent.")
    parser.add_argument('--actor_params', type=str,
                        default='checkpoint_actor.pth',
                        help="Filepath of the actor-network-parameters to \
                              be used in the agent's model to run the game.")
    parser.add_argument('--critic_params', type=str,
                        default='checkpoint_critic.pth',
                        help="Filepath of the critic-network-parameters to \
                              be used in the agent's model to run the game.")
    parser.add_argument('--watch', action='store_true',
                        help='Watch agent during training.')

    args = parser.parse_args()
    
    # Run the game with a smart agent
    if args.action == 'run':
        run_model(args.actor_params, args.critic_params)
    # Train an agent
    elif args.action == 'train':
        scores = train_agent(args.watch)
        # Plot rewards
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        print('Unkown action! Possible actions are "run" and "train".')