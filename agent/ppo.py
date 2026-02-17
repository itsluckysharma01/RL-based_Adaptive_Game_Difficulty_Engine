import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    """Actor-Critic Network for PPO"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        action_probs = torch.softmax(self.actor(x), dim=-1)
        state_value = self.critic(x)
        
        return action_probs, state_value


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_size, action_size, config):
        """
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Dictionary with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.epsilon_clip = config.get('epsilon_clip', 0.2)
        self.k_epochs = config.get('k_epochs', 4)
        self.learning_rate = config.get('learning_rate', 0.0003)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Network
        hidden_size = config.get('hidden_size', 128)
        self.policy = ActorCritic(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
    
    def select_action(self, state, training=True):
        """Select action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        if training:
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(action_log_prob.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def train(self):
        """Update policy using collected experiences"""
        if len(self.states) == 0:
            return None
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # Calculate returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update
        total_loss = 0
        for _ in range(self.k_epochs):
            # Get current policy predictions
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            # Calculate advantages
            advantages = returns - state_values.squeeze()
            
            # Calculate ratio and surrogate loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages.detach()
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages.detach()
            
            # Total loss
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.value_loss_coef * (advantages.pow(2)).mean()
            entropy_loss = -self.entropy_coef * entropy.mean()
            
            loss = actor_loss + critic_loss + entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.clear_memory()
        
        return total_loss / self.k_epochs
    
    def clear_memory(self):
        """Clear stored experiences"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
    
    def save(self, filepath):
        """Save model to file"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])