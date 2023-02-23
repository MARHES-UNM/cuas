import math
import os
import pathlib
import numpy as np
from cuas import util
import math
from enum import IntEnum

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# RESOURCE_FOLDER = pathlib.Path("resources")
RESOURCE_FOLDER = path.joinpath("../../resources")


class AgentType(IntEnum):
    P = 0  # pursuer
    E = 1  # evader
    O = 2  # obstacle
    # T = 3  # Target


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class ObsPaddingType(IntEnum):
    last = 0
    zeros = 1
    max = 2
    min = 3


# TODO: move env_width and height to cuas_agents
class Entity:
    """Defines an entity for the simulation. All units are in metric"""

    def __init__(self, x, y, r=1.5, env_width=80, env_height=60, type=AgentType.P):
        self.x = x
        self.y = y
        self.radius = r
        self.env_width = env_width
        self.env_height = env_height
        self._in_collision = False
        self._type = type
        self._state = np.array([self.x, self.y, self.radius])

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, type):
        self._type = type

    @property
    def pos(self):
        """Return agent's position

        Returns:
            _type_: _description_
        """
        return np.array([self.x, self.y])

    def rel_dist(self, entity):
        """Calculates relative distance to another object"""
        dist = util.distance((self.x, self.y), (entity.x, entity.y))

        return dist

    def rel_bearing(self, entity):
        """Calculates relative bearing to another object"""
        bearing = util.angle((self.x, self.y), (entity.x, entity.y))

        return bearing

    def rel_bearing_error(self, entity):
        """[summary]

        Args:
            entity ([type]): [description]

        Returns:
            [type]: [description]
        """
        bearing = util.angle((self.x, self.y), (entity.x, entity.y)) - self.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing

    def rel_bearing_entity_error(self, entity):
        bearing = util.angle((self.x, self.y), (entity.x, entity.y)) - entity.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing

    def collision_with(self, entity):
        """Returns if has collided with another entity"""
        in_collision = False

        rel_dist_to_entity = self.rel_dist(entity)

        if rel_dist_to_entity < (self.radius + entity.radius):
            in_collision = True
            self._in_collision = in_collision

        return in_collision


class Agent(Entity):
    """Defines an agent for the simulation. All units are in metric, time is second

    Args:
        Entity ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self, x, y, theta, r=1.5, type=AgentType.P, obs_r=100
    ):  # assume env size 80x60
        super().__init__(x, y, r=r, type=type)
        self.theta = theta  # in radians
        self.dt = 0.01  # 10 ms
        self.v = 0
        self.w = 0
        self.obs_r = obs_r

    # TODO: this should just return the states not the state variable
    @property
    def state(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return np.array([self.x, self.y, self.theta, self.v, self.w])

    def step(self, action):
        """Updates the agent's state. Positive radians is counterclockwise"""
        v = action[0]
        w = action[1]

        if w >= 0.01 or w <= -0.01:
            ratio = v / w
            self.x += -ratio * math.sin(self.theta) + ratio * math.sin(
                self.theta + w * self.dt
            )
            self.y += ratio * math.cos(self.theta) - ratio * math.cos(
                self.theta + w * self.dt
            )
            self.theta += w * self.dt

        else:
            self.x += v * self.dt * math.cos(self.theta)
            self.y += v * self.dt * math.sin(self.theta)
            self.theta += w * self.dt

        self.v = action[0]
        self.w = action[1]

        self._check_bounds()

    def _check_bounds(self):
        """Checks bound for moving objects"""

        self.x = min(max(0, self.x), self.env_width)
        self.y = min(max(0, self.y), self.env_height)

        # see: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
        # self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        # wrap theta to -pi and pi
        # This is more efficient than using np.arctan2
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # TODO: fix this

    def sensed(self, entity):
        """Returns true if agent can sense entity
        # https://github.com/Attila94/EKF-SLAM/blob/master/robot.py
        # https://www.geeksforgeeks.org/check-whether-point-exists-circle-sector-not/
        # https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector
        """

        _detected = False
        rel_bearing = self.rel_bearing(entity)
        rel_distance = self.rel_dist(entity)
        return True if rel_distance < self.obs_r else False

    def uni_to_si_dyn(self, dxu, projection_distance=0.05):
        """
        See:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py

        """
        cs = np.cos(self.theta)
        ss = np.sin(self.theta)

        dxi = np.zeros(2)
        dxi[0] = cs * dxu[0] - projection_distance * ss * dxu[1]
        dxi[1] = ss * dxu[0] + projection_distance * cs * dxu[1]

        return dxi

    def si_to_uni_dyn(self, si_v, projection_distance=0.05):
        """
        see:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py
        also:
            https://arxiv.org/pdf/1802.07199.pdf

        Args:
            agent ([type]): [description]
            si_v ([type]): [description]
            projection_distance (float, optional): [description]. Defaults to 0.05.

        Returns:
            [type]: [description]
        """
        cs = np.cos(self.theta)
        ss = np.sin(self.theta)

        dxu = np.zeros(2)
        dxu[0] = cs * si_v[0] + ss * si_v[1]
        dxu[1] = (1 / projection_distance) * (-ss * si_v[0] + cs * si_v[1])

        return dxu


class Agent2D(Agent):
    def __init__(
        self, x, y, theta, r=1.5, type=AgentType.P, obs_r=100
    ):  # assume env size 80x60
        super().__init__(x=x, y=y, theta=theta, r=r, type=type, obs_r=obs_r)
        self.dt = 0.01  # 10 ms
        self.vx = 0
        self.vy = 0

    @property
    def state(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return np.array([self.x, self.y, self.theta, self.vx, self.vy])

    def step(self, action):
        """Updates the agent's state. Positive radians is counterclockwise"""
        self.vx = action[0]
        self.vy = action[1]

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.theta = np.arctan2(self.vy, self.vx)

        self._check_bounds()


class Obstacle(Agent):
    """Defines and obstacle in the environment
    Args:
        Agent ([type]): [description]
    """

    def __init__(
        self,
        x,
        y,
        theta,
        r=1.5,
        type=AgentType.O,
        obs_type=ObsType.S,
        v_min=0,
        v_max=2,
        w_min=-np.pi,
        w_max=np.pi,
    ):
        super().__init__(x, y, theta, r, type=type)

        self._obs_type = obs_type
        self._v_min = v_min
        self._v_max = v_max
        self._w_min = w_min
        self._w_max = w_max

    def step(self):
        if self._obs_type == ObsType.S:
            action = [0, 0]

        else:
            # get random v and random w
            v = np.random.uniform(low=self._v_min, high=self._v_max)
            w = np.random.uniform(low=self._w_min, high=self._w_max)
            action = [v, w]

        super().step(action)

    # @overload
    # def _check_bounds(self):
    #     # min_x = -self.radius / 2
    #     # min_y = -self.radius / 2
    #     # max_x = self.env_width + self.radius / 2
    #     # max_y = self.env_height + self.radius / 2
    #     min_x = 0
    #     min_y = 0
    #     max_x = self.env_width
    #     max_y = self.env_height

    #     if self.x < min_x:
    #         self.x = max_x
    #     elif self.x > max_x:
    #         self.x = min_x
    #     if self.y < min_y:
    #         self.y = max_y
    #     elif self.y > max_x:
    #         self.y = min_y

    #     self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi


class CuasAgent(Agent):
    """[summary]

    Args:
        Agent ([type]): [description]
    """

    def __init__(self, id, type, x, y, theta, r=1.5, obs_r=100, move_type="rl"):
        super().__init__(x, y, theta, r=r, obs_r=obs_r, type=type)
        self.id = id
        self.done = False
        self.reported_done = False
        self.move_type = move_type
        self.captured = False


class CuasAgent2D(Agent2D):
    """[summary]

    Args:
        Agent ([type]): [description]
    """

    def __init__(self, id, type, x, y, theta, r=1.5, obs_r=100, move_type="rl"):
        super().__init__(x, y, theta, r=r, obs_r=obs_r, type=type)
        self.id = id
        self.done = False
        self.reported_done = False
        self.move_type = move_type
        self.captured = False