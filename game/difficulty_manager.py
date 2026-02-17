class DifficultyManager:
    """Manages game difficulty based on RL agent actions"""
    
    def __init__(self, initial_speed=10):
        """
        Args:
            initial_speed: Starting game speed (FPS)
        """
        self.current_speed = initial_speed
        self.min_speed = 5
        self.max_speed = 20
        self.speed_delta = 2
        self.difficulty_level = 1  # 0=easy, 1=medium, 2=hard
    
    def apply_action(self, action, game):
        """
        Apply difficulty adjustment action to the game
        
        Args:
            action: 0=increase difficulty, 1=decrease difficulty, 2=no change
            game: SnakeGame instance
        """
        if action == 0:  # Increase difficulty
            self.current_speed = min(self.current_speed + self.speed_delta, self.max_speed)
            self.difficulty_level = 2
        elif action == 1:  # Decrease difficulty
            self.current_speed = max(self.current_speed - self.speed_delta, self.min_speed)
            self.difficulty_level = 0
        else:  # No change
            self.difficulty_level = 1
        
        # Apply speed to game clock
        return self.current_speed
    
    def get_difficulty_level(self):
        """Return current difficulty level"""
        return self.difficulty_level
    
    def reset(self):
        """Reset to initial difficulty"""
        self.current_speed = 10
        self.difficulty_level = 1