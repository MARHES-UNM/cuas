import math
import gym
from gym import spaces
import numpy as np

RAD2DEG = 57.29577951308232
DEG2RAD = 0.017453292519943295


def cartesian2polar(point1=(0, 0), point2=(0, 0)):
    """ Retuns conversion of cartesian to polar coordinates """
    r = distance(point1, point2)
    alpha = angle(point1, point2)

    return r, alpha


def distance(point_1=(0, 0), point_2=(0, 0)):
    """Returns the distance between two points"""
    return math.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)


def angle(point_1=(0, 0), point_2=(0, 0)):
    """Returns the angle between two points"""
    return math.atan2(point_2[1] - point_1[1], point_2[0] - point_1[0])


def angle_2pi(point_1=(0, 0), point_2=(0, 0)):
    """Returns the angle between two points"""
    return (
        math.atan2(point_2[1] - point_1[1], (point_2[0] - point_1[0])) + 2 * math.pi
    ) % (2 * math.pi)


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width / 2
    image.anchor_y = image.height / 2


def norm_data(data, max, min):
    """
    returns data normalize to range [-1, 1]

    Args:
        data ([type]): [description]
        max ([type]): [description]
        min ([type]): [description]
    """
    norm_data = (data - min) / (max - min)
    norm_data = 2 * norm_data - 1

    return norm_data


class RescaleAction(gym.ActionWrapper):
    # why normalize env
    """Rescales the continuous action space of the environment to a range [a,b].
    Example::
        >>> RescaleAction(env, a, b).action_space == Box(a,b)
        True
    """

    def __init__(self, env, a, b):
        assert isinstance(
            env.action_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.action_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action


class NormalizedObservation(gym.ObservationWrapper):
    def __init__(self, env, a, b):
        assert isinstance(
            env.observation_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.observation_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(NormalizedObservation, self).__init__(env)
        self.a = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + a
        )
        self.b = (
            np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype) + b
        )
        self.observation_space = spaces.Box(
            low=a,
            high=b,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        assert np.all(np.greater_equal(observation, self.env.observation_space.low)), (
            observation,
            self.env.observation_space.low,
        )
        assert np.all(np.less_equal(observation, self.env.observation_space.high)), (
            observation,
            self.env.observation_space.high,
        )

        low = self.env.observation_space.low
        high = self.env.observation_space.high

        observation = (observation - low) / (high - low)
        observation = np.clip(observation, self.a, self.b)
        return observation
