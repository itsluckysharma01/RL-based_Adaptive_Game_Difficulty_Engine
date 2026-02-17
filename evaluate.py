import yaml
import numpy as np
import pygame
from game.snake import SnakeGame
from game.metrics import GameMetrics
from game.difficulty_manager import DifficultyManager
from agent.dqn import DQNAgent
from agent.ppo import PPOAgent


def evaluate_agent(agent_type='dqn', model_path='models/dqn_final.pth', 
                   config_path='config/hyperparameters.yaml', episodes=10):
    """
    Evaluate trained agent
    
    Args:
        agent_type: 'dqn' or 'ppo'
        model_path: Path to saved model
        config_path: Path to config file
        episodes: Number of episodes to evaluate
    """
    print(f"Evaluating {agent_type.upper()} Agent...")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    env_config = config['environment']
    
    # Initialize agent
    if agent_type == 'dqn':
        agent_config = config['dqn']
        agent_config['epsilon_start'] = 0.0  # No exploration during evaluation
        agent = DQNAgent(
            state_size=env_config['state_size'],
            action_size=env_config['action_size'],
            config=agent_config
        )
    else:
        agent_config = config['ppo']
        agent = PPOAgent(
            state_size=env_config['state_size'],
            action_size=env_config['action_size'],
            config=agent_config
        )
    
    # Load trained model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    
    scores = []
    survival_times = []
    difficulty_changes = []
    
    for episode in range(episodes):
        # Initialize environment
        game = SnakeGame()
        game.display = pygame.display.set_mode((600, 400))
        pygame.display.set_caption(f"Evaluation - Episode {episode + 1}/{episodes}")
        
        metrics = GameMetrics()
        difficulty_manager = DifficultyManager()
        
        step = 0
        actions_taken = []
        
        game.reset()
        
        while not game.game_over and step < 1000:
            # Get current state
            state = [
                game.score,
                int(pygame.time.get_ticks() / 1000),
                0,
                difficulty_manager.get_difficulty_level()
            ]
            
            # Select action (no exploration)
            action = agent.select_action(state, training=False)
            actions_taken.append(action)
            
            # Apply difficulty adjustment
            speed = difficulty_manager.apply_action(action, game)
            
            # Game step
            game.handle_input()
            game.move_snake()
            game.draw()
            game.clock.tick(speed)
            
            metrics.update_score(game.score)
            step += 1
        
        scores.append(game.score)
        survival_times.append(metrics.get_survival_time())
        difficulty_changes.append(len(set(actions_taken)))
        
        print(f"Episode {episode + 1}: Score={game.score}, "
              f"Survival Time={metrics.get_survival_time()}s, "
              f"Unique Actions={len(set(actions_taken))}")
        
        pygame.quit()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Best Score: {np.max(scores)}")
    print(f"Average Survival Time: {np.mean(survival_times):.2f}s ± {np.std(survival_times):.2f}s")
    print(f"Average Difficulty Changes: {np.mean(difficulty_changes):.2f}")
    print("="*50)
    
    return scores, survival_times


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        agent_type = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else f'models/{agent_type}_final.pth'
        evaluate_agent(agent_type=agent_type, model_path=model_path)
    else:
        print("Usage: python evaluate.py <dqn|ppo> [model_path]")
        print("Example: python evaluate.py dqn models/dqn_final.pth")