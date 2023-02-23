import gym
from gym import error, spaces, utils
from gym.utils import seeding


class BaseCuasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()

    def step(self, action):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self, mode="human"):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()