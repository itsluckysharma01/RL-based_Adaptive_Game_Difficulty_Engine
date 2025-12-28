import time

class GameMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.score = 0
        self.deaths = 0
        self.difficulty_level = 0

    def update_score(self, score):
        self.score = score

    def record_death(self):
        self.deaths += 1

    def get_survival_time(self):
        return int(time.time() - self.start_time)

    def get_state(self):
        return [
            self.score,
            self.get_survival_time(),
            self.deaths,
            self.difficulty_level
        ]
