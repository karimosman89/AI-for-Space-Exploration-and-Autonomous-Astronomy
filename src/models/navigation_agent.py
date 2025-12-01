"""
Autonomous Navigation Agent using Reinforcement Learning

This module implements a Deep Q-Network (DQN) and PPO-based agent
for autonomous spacecraft navigation and trajectory optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import random


class NavigationNetwork(nn.Module):
    """Neural network for Q-value or policy estimation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(NavigationNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class NavigationAgent:
    """
    Reinforcement Learning agent for autonomous spacecraft navigation.
    
    Features:
    - Deep Q-Learning for discrete action spaces
    - Experience replay for stable learning
    - Target network for stable Q-value estimation
    - Epsilon-greedy exploration strategy
    """
    
    def __init__(
        self,
        state_dim: int = 12,  # position, velocity, orientation, fuel
        action_dim: int = 6,  # thrust directions + rotation
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10
    ):
        """
        Initialize the navigation agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Q-networks
        self.policy_net = NavigationNetwork(state_dim, action_dim)
        self.target_net = NavigationNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Training statistics
        self.steps_done = 0
        self.episode_rewards = []
        
        # Action mapping
        self.actions = [
            'thrust_forward', 'thrust_backward',
            'thrust_left', 'thrust_right',
            'thrust_up', 'thrust_down'
        ]
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether in training mode (enables exploration)
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: best action according to policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(1).item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Returns:
            Loss value if training performed, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.smooth_l1_loss(
            current_q_values.squeeze(),
            target_q_values
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
        
        return loss.item()
    
    def plan_trajectory(
        self,
        start_state: np.ndarray,
        goal_state: np.ndarray,
        max_steps: int = 1000
    ) -> Tuple[List[int], List[np.ndarray], float]:
        """
        Plan trajectory from start to goal state.
        
        Args:
            start_state: Initial state
            goal_state: Desired goal state
            max_steps: Maximum planning steps
            
        Returns:
            Tuple of (action sequence, state sequence, total reward)
        """
        self.policy_net.eval()
        
        actions = []
        states = [start_state]
        total_reward = 0.0
        
        current_state = start_state.copy()
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(current_state, training=False)
            actions.append(action)
            
            # Simulate environment step (simplified)
            next_state, reward, done = self._simulate_step(
                current_state,
                action,
                goal_state
            )
            
            states.append(next_state)
            total_reward += reward
            
            if done:
                break
            
            current_state = next_state
        
        return actions, states, total_reward
    
    def _simulate_step(
        self,
        state: np.ndarray,
        action: int,
        goal: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Simulate one environment step (simplified dynamics).
        
        Args:
            state: Current state
            action: Action to take
            goal: Goal state
            
        Returns:
            Tuple of (next_state, reward, done)
        """
        # Simplified spacecraft dynamics
        next_state = state.copy()
        
        # Action effects (simplified)
        action_effects = {
            0: np.array([0.1, 0, 0]),    # thrust_forward
            1: np.array([-0.1, 0, 0]),   # thrust_backward
            2: np.array([0, 0.1, 0]),    # thrust_left
            3: np.array([0, -0.1, 0]),   # thrust_right
            4: np.array([0, 0, 0.1]),    # thrust_up
            5: np.array([0, 0, -0.1]),   # thrust_down
        }
        
        # Update position (first 3 dimensions)
        if action in action_effects:
            next_state[:3] += action_effects[action]
        
        # Calculate reward based on distance to goal
        current_distance = np.linalg.norm(state[:3] - goal[:3])
        next_distance = np.linalg.norm(next_state[:3] - goal[:3])
        
        reward = current_distance - next_distance  # Reward for getting closer
        reward -= 0.01  # Small penalty for fuel consumption
        
        # Check if goal reached
        done = next_distance < 0.1
        if done:
            reward += 100.0  # Large bonus for reaching goal
        
        return next_state, reward, done
    
    def save(self, filepath: str):
        """Save agent's neural networks."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent's neural networks."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


if __name__ == "__main__":
    # Example usage
    print("Navigation Agent initialized")
    agent = NavigationAgent(state_dim=12, action_dim=6)
    
    # Test trajectory planning
    start = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # position + velocity + fuel
    goal = np.array([10, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    actions, states, reward = agent.plan_trajectory(start, goal, max_steps=100)
    print(f"\nPlanned trajectory:")
    print(f"Actions: {len(actions)}")
    print(f"Total reward: {reward:.2f}")
    print(f"Final state: {states[-1][:3]}")
