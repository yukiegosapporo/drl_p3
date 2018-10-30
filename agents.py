
import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3       # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
# EPSILON = 1.0
# EPSILON_DECAY = 1e-6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agents():
    def __init__(self, state_size, action_size, random_seed, num_agents, train_mode, agent):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.agent = agent
        self.agents = [self.agent(state_size, action_size, random_seed, num_agents, self.memory, idx, train_mode) for idx in range(self.num_agents)]
        self.train_mode = train_mode
    
    # def __repl__(self):
    #     return 

    def step(self, states, actions, rewards, next_states, dones, timestep):
        if self.train_mode:
            rewards = np.reshape(rewards, self.num_agents, -1)
            dones = np.reshape(dones, self.num_agents, -1)
            self.memory.add(states, actions, rewards, next_states, dones)
            if len(self.memory) > BATCH_SIZE and timestep % 1 == 0:
                for _ in range(1):
                    experiences = self.memory.sample()
                    # if self.agent == SharedStateAgent:
                    #     states, actions, rewards, next_states, dones = experiences
                    #     actions_next, actions_pred = self.call_actors(states, next_states)
                    for idx, agent in enumerate(self.agents):
                        agent.learn(experiences, GAMMA, idx)
                        # if self.agent == Agent:
                        #     agent.learn(experiences, GAMMA, idx)
                        # elif self.agent == SharedStateAgent:
                        #     agent.learn(experiences, GAMMA, idx, actions_next, actions_pred)

    # def call_actors(self, states, next_states):
    #     """
    #     gather actions from all the agents
    #     """
    #     actions_pred = []
    #     actions_next = []
    #     for i, agent in enumerate(self.agents):
    #         state = states.reshape(-1, 2, self.state_size).index_select(1, torch.tensor([i])).squeeze(1)
    #         next_state = next_states.reshape(-1, 2, self.state_size).index_select(1, torch.tensor([i])).squeeze(1)
    #         actions_next.append(agent.actor_target(next_state))
    #         actions_pred.append(agent.actor_local(state))
    #     return torch.cat(actions_next, 1), torch.cat(actions_pred, 1)

    def act(self, states, noise_weight=1.0, add_noise=True):
        actions = []
        for idx, agent in enumerate(self.agents):
            if self.agent == SharedMemoryAgent:
                action = agent.act(states[idx], noise_weight, add_noise)
            elif self.agent == SharedStateAgent:
                states = np.reshape(states, -1)
                action = agent.act(states, noise_weight, add_noise)
            actions.append(action)
        return np.array(actions)

    def reset(self):
        for agent in self.agents:
            agent.noise.reset()
    
    def soft_update(self, local_model, target_model, tau):
        for agent in self.agents:
            agent.soft_update(local_model, target_model, tau)
    
    def hard_update(self, target, source):
        for agent in self.agents:
            agent.hard_update(target, source)
    
    def noise_reset(self):
        for agent in self.agents:
            agent.noise.reset()

class SharedMemoryAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, num_agents, Memory, id, train_mode):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.id = id
        self.train_mode = train_mode

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # self.criticLoss = nn.MSELoss()
        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        # self.epsilon = EPSILON

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.memory = Memory
    
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)


    # def step(self, states, actions, rewards, next_states, dones, timestep):
    #     """Save experience in replay memory, and use random sample from buffer to learn."""
    #     # Save experience / reward
    #     for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
    #         self.memory.add(state, action, reward, next_state, done)
    #     # Learn, if enough samples are available in memory
    #     if len(self.memory) > BATCH_SIZE and timestep % 20 == 0:
    #         for _ in range(10):
    #             experiences = self.memory.sample()
    #             self.learn(experiences, GAMMA)


    def act(self, states, noise_weight=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # if self.agent = SharedStateAgent:
            #     states = np.reshape(states, -1)
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        add_noise = self.train_mode
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, i):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            t: time step
        """
        states, actions, rewards, next_states, dones = experiences
        
        agent_id = torch.tensor([i]).to(device)
        states = states.reshape(-1, 2, self.state_size).index_select(1, agent_id).squeeze(1)
        actions = actions.reshape(-1, 2, self.action_size).index_select(1, agent_id).squeeze(1)
        rewards = rewards.reshape(-1, 2, 1).index_select(1, agent_id).squeeze(1)
        dones = dones.reshape(-1, 2, 1).index_select(1, agent_id).squeeze(1)
        next_states = next_states.reshape(-1, 2, self.state_size).index_select(1, agent_id).squeeze(1)
        # print(states.shape, actions.shape, rewards.shape, dones.shape, next_states.shape)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)   

        # ----------------------- update noise ----------------------- #        
        self.noise.reset()


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
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
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SharedStateAgent(SharedMemoryAgent):
    def __init__(self, state_size, action_size, random_seed, num_agents, Memory, id, train_mode):
        super().__init__(state_size, action_size, random_seed, num_agents, Memory, id, train_mode)
        self.actor_local = Actor(state_size * self.num_agents, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size * self.num_agents, action_size, random_seed).to(device)
        self.critic_local = Critic(state_size * self.num_agents, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size * self.num_agents, action_size, random_seed).to(device)

    def learn(self, experiences, gamma, i,):
        states, actions, rewards, next_states, dones = experiences
        
        agent_id = torch.tensor([i]).to(device)
        # state_i = states.reshape(-1, 2, self.state_size).index_select(1, agent_id).squeeze(1)
        states = states.reshape(-1, 1, self.state_size * self.num_agents).squeeze(1)
        # actions_flattened = actions.reshape(-1, 1, self.action_size * self.num_agents).squeeze(1)
        actions = actions.reshape(-1, 2, self.action_size).index_select(1, agent_id).squeeze(1)
        rewards = rewards.reshape(-1, 2, 1).index_select(1, agent_id).squeeze(1)
        dones = dones.reshape(-1, 2, 1).index_select(1, agent_id).squeeze(1)
        # next_state_i = next_states.reshape(-1, 2, self.state_size).index_select(1, agent_id).squeeze(1)
        next_states = next_states.reshape(-1, 1, self.state_size * self.num_agents).squeeze(1)

        # actions_next_i = actions_next.reshape(-1, 2, 2).index_select(1, agent_id).squeeze(1)
        # actions_pred_i = actions_pred.reshape(-1, 2, 2).index_select(1, agent_id).squeeze(1)
        # print(states.shape, actions.shape, rewards.shape, dones.shape, next_states.shape)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_state_i)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)   

        # ----------------------- update noise ----------------------- #        
        self.noise.reset()
    