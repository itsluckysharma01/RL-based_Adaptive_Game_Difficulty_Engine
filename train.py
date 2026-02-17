import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from game.snake import SnakeGame
from game.metrics import GameMetrics
from game.difficulty_manager import DifficultyManager
from agent.dqn import DQNAgent
from agent.ppo import PPOAgent
from agent.replay_buffer import ReplayBuffer
import pygame

def calculate_reward(metrics, game_over, prev_score, prev_time):
    """
    Calculate reward based on player performance
    Reward structure:
    - Positive reward for scoring
    - Positive reward for survival
    - Negative reward for dying
    """
    reward = 0
    
    # Score increase
    score_increase = metrics.score - prev_score
    reward += score_increase * 10  # +10 per food eaten
    
    # Survival bonus
    time_increase = metrics.get_survival_time() - prev_time
    reward += time_increase * 0.1  # Small bonus for staying alive
    
    # Death penalty
    if game_over:
        reward -= 50
    
    return reward


def train_dqn(config_path='config/hyperparameters.yaml', save_dir='models'):
    """Train DQN agent"""
    print("Starting DQN Training...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dqn_config = config['dqn']
    train_config = config['training']
    env_config = config['environment']
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize components
    agent = DQNAgent(
        state_size=env_config['state_size'],
        action_size=env_config['action_size'],
        config=dqn_config
    )
    replay_buffer = ReplayBuffer(capacity=dqn_config['memory_size'])
    
    # Training stats
    episode_rewards = []
    episode_scores = []
    losses = []
    
    for episode in range(train_config['episodes']):
        # Initialize environment
        game = SnakeGame()
        game.display = pygame.display.set_mode((600, 400))
        pygame.display.set_caption(f"DQN Training - Episode {episode + 1}")
        
        metrics = GameMetrics()
        difficulty_manager = DifficultyManager()
        
        total_reward = 0
        prev_score = 0
        prev_time = 0
        step = 0
        
        game.reset()
        
        while not game.game_over and step < train_config['max_steps']:
            # Get current state
            state = [
                game.score,
                int(pygame.time.get_ticks() / 1000),
                0,  # deaths (per episode)
                difficulty_manager.get_difficulty_level()
            ]
            
            # Select action
            action = agent.select_action(state, training=True)
            
            # Apply difficulty adjustment
            speed = difficulty_manager.apply_action(action, game)
            
            # Game step
            game.handle_input()
            game.move_snake()
            if train_config.get('render', False):
                game.draw()
            game.clock.tick(speed)
            
            # Get next state and reward
            next_state = [
                game.score,
                int(pygame.time.get_ticks() / 1000),
                1 if game.game_over else 0,
                difficulty_manager.get_difficulty_level()
            ]
            
            reward = calculate_reward(
                metrics, 
                game.game_over, 
                prev_score,
                prev_time
            )
            metrics.update_score(game.score)
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, game.game_over)
            
            # Train agent
            loss = agent.train(replay_buffer)
            if loss is not None:
                losses.append(loss)
            
            total_reward += reward
            prev_score = game.score
            prev_time = int(pygame.time.get_ticks() / 1000)
            step += 1
        
        episode_rewards.append(total_reward)
        episode_scores.append(game.score)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            avg_loss = np.mean(losses[-100:]) if losses else 0
            print(f"Episode {episode + 1}/{train_config['episodes']} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f} | "
                  f"Loss: {avg_loss:.4f} | Epsilon: {agent.epsilon:.3f}")
        
        # Save model
        if (episode + 1) % train_config['save_frequency'] == 0:
            agent.save(f"{save_dir}/dqn_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
        
        pygame.quit()
    
    # Save final model
    agent.save(f"{save_dir}/dqn_final.pth")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_scores, losses, 'DQN')
    
    print("DQN Training Complete!")
    return agent


def train_ppo(config_path='config/hyperparameters.yaml', save_dir='models'):
    """Train PPO agent"""
    print("Starting PPO Training...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    ppo_config = config['ppo']
    train_config = config['training']
    env_config = config['environment']
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Initialize agent
    agent = PPOAgent(
        state_size=env_config['state_size'],
        action_size=env_config['action_size'],
        config=ppo_config
    )
    
    # Training stats
    episode_rewards = []
    episode_scores = []
    losses = []
    
    for episode in range(train_config['episodes']):
        # Initialize environment
        game = SnakeGame()
        game.display = pygame.display.set_mode((600, 400))
        pygame.display.set_caption(f"PPO Training - Episode {episode + 1}")
        
        metrics = GameMetrics()
        difficulty_manager = DifficultyManager()
        
        total_reward = 0
        prev_score = 0
        prev_time = 0
        step = 0
        
        game.reset()
        
        while not game.game_over and step < train_config['max_steps']:
            # Get current state
            state = [
                game.score,
                int(pygame.time.get_ticks() / 1000),
                0,
                difficulty_manager.get_difficulty_level()
            ]
            
            # Select action
            action = agent.select_action(state, training=True)
            
            # Apply difficulty adjustment
            speed = difficulty_manager.apply_action(action, game)
            
            # Game step
            game.handle_input()
            game.move_snake()
            if train_config.get('render', False):
                game.draw()
            game.clock.tick(speed)
            
            # Calculate reward
            reward = calculate_reward(
                metrics,
                game.game_over,
                prev_score,
                prev_time
            )
            metrics.update_score(game.score)
            
            # Store transition
            agent.store_transition(reward, game.game_over)
            
            total_reward += reward
            prev_score = game.score
            prev_time = int(pygame.time.get_ticks() / 1000)
            step += 1
        
        # Train agent after episode
        loss = agent.train()
        if loss is not None:
            losses.append(loss)
        
        episode_rewards.append(total_reward)
        episode_scores.append(game.score)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            avg_loss = np.mean(losses[-10:]) if losses else 0
            print(f"Episode {episode + 1}/{train_config['episodes']} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f} | "
                  f"Loss: {avg_loss:.4f}")
        
        # Save model
        if (episode + 1) % train_config['save_frequency'] == 0:
            agent.save(f"{save_dir}/ppo_episode_{episode + 1}.pth")
            print(f"Model saved at episode {episode + 1}")
        
        pygame.quit()
    
    # Save final model
    agent.save(f"{save_dir}/ppo_final.pth")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_scores, losses, 'PPO')
    
    print("PPO Training Complete!")
    return agent


def plot_training_results(rewards, scores, losses, agent_name):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Rewards
    axes[0].plot(rewards)
    axes[0].set_title(f'{agent_name} - Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].grid(True)
    
    # Scores
    axes[1].plot(scores)
    axes[1].set_title(f'{agent_name} - Episode Scores')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].grid(True)
    
    # Losses
    axes[2].plot(losses)
    axes[2].set_title(f'{agent_name} - Training Loss')
    axes[2].set_xlabel('Training Step')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{agent_name.lower()}_training_results.png')
    print(f"Training plots saved to plots/{agent_name.lower()}_training_results.png")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'ppo':
        train_ppo()
    else:
        train_dqn()