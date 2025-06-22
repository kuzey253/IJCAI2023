import random

class random_agent:
    def __init__(self):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]

        self.buffer = []
        self.counter = 0

    def select_action(self, obs, train=False):
        return self.act(obs)

    def act(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [force, angle]
    
    def store_transition(self, transition):
        # Random agent does not store transitions
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # Random agent does not update
        pass







