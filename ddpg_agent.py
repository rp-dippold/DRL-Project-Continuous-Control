import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from config import Config

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, add_noise):
        """Initialize an Agent object.
        
        Parameters
        ----------
        state_size : int
            Dimension of each state
        action_size : int
            Dimension of each action
        add_noise : boolean
            Add noise to actions for exploration
        """
        config = Config.get_config()
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.tau = config.tau # interpolation parameter
        self.gamma = config.gamma
        self.update_every = config.update_every
        self.network_update = config.network_update
        self.seed = np.random.seed(config.random_seed)
        self.add_noise = add_noise
        if config.device == 'GPU' and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Actor Network (with Target Network)
        self.actor_local = Actor(state_size, action_size,
                                 config.random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size,
                                  config.random_seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), 
                                          lr=config.lr_actor)

        # Critic Network (with Target Network)
        self.critic_local = Critic(state_size, action_size,
                                   config.random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size,
                                    config.random_seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), 
                                           lr=config.lr_critic, 
                                           weight_decay=config.weight_decay)

        # Noise process
        if self.add_noise:
            self.noise = OUNoise(action_size, config.random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size,
                                   self.batch_size, config.random_seed,
                                   self.device)


    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer 
           to learn.
        """
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn after every UPDATE_EVERY time steps and if enough samples 
        # are available in memory
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            # update the network NETWORK_UPDATE times
            for _ in range(self.network_update):
                experiences = self.memory.sample()
                self.learn(experiences)


    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if self.add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        """Rest noise object."""
        self.noise.reset()


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience
        tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
        experiences : (Tuple[torch.Tensor])
            tuple of (s, a, r, s', done) tuples 
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ----------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients during training to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ------------------------ #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     


    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Parameters
        ----------
        local_model : PyTorch model 
            Weights will be copied from
        target_model : PyTorch model
            Weights will be copied to
        """
        for target_param, local_param in \
            zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + \
                                    (1.0-self.tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed):
        """Initialize parameters and noise process."""
        config = Config.get_config()
        self.mu = config.ou_mu * np.ones(size)
        self.theta = config.ou_theta
        self.sigma = config.ou_sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Parameters
        ----------
        action_size : int 
            Dimension of action
        buffer_size : int 
            Maximum size of buffer
        batch_size : int
            Size of each training batch
        seed : int 
            Seed to initialize random generator
        device: string
            GPU or CPU
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size) # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"]
        )
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack(
            [e.state for e in experiences if e is not None])).float().to(
                self.device)
        actions = torch.from_numpy(np.vstack(
            [e.action for e in experiences if e is not None])).float().to(
                self.device)
        rewards = torch.from_numpy(np.vstack(
            [e.reward for e in experiences if e is not None])).float().to(
                self.device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(
                self.device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(
                np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)