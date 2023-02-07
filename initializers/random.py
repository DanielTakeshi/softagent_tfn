import random

from initializers import InitializerBase

class RandomInitializer(InitializerBase):
    def __init__(self, env, args):
        self.env = env

    def get_action(self, obs, info=None):
        action = self.env.action_space.sample()

        # done with probability 0.5
        done = random.random() < 0.5

        return action, done
