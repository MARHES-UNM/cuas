import gym
from gym import error, spaces, utils
from gym.utils import seeding
from cuas.envs.base_cuas_env import BaseCuasEnv
import numpy as np
import math
from cuas import util
import os

import pathlib

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
# RESOURCE_FOLDER = pathlib.Path("resources")
RESOURCE_FOLDER = path.joinpath("../../resources")


class EntityState:
    """Defines an entity for the simulation. All units are in metric"""

    def __init__(self, x, y, theta, width=1.5, height=1.5):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width
        self.height = height

    def get_state(self):
        return [self.x, self.y, self.theta]


class Agent(EntityState):
    def __init__(self, x, y, theta):
        EntityState.__init__(self, x, y, theta)
        self.dt = 0.01  # 10 ms
        self.v = 0
        self.omega = 0

    def update_state(self, v, omega):
        self.v = v
        self.omega = omega
        self.x += v * math.cos(self.theta) * self.dt
        self.y += v * math.sin(self.theta) * self.dt
        self.theta += omega * self.dt

    def unicycle_to_si_state(self):
        pass

    def si_input_to_unicycle(self):
        """Converts single integrator input to unicycle v and omega"""
        pass

    def distance_to_entity(self, entity):
        dist = util.distance((self.x, self.y), (entity.x, entity.y))

        return dist

    def bearing_to_entity(self, entity):
        bearing = util.angle((self.x, self.y), (entity.x, entity.y))

        return bearing

    def get_state(self):
        return [self.x, self.y, self.theta, self.v, self.omega]

    def attacker_neutralized(self, attacker):
        distance_to_attacker = self.distance_to_entity(attacker)

        if distance_to_attacker < attacker.neutralize_radius:
            attacker.neutralized = True

        return attacker.neutralized


class Attacker(Agent):
    def __init__(self, x, y, theta):
        Agent.__init__(self, x, y, theta)
        self.neutralized = False
        self.neutralize_radius = self.width * 1.2

        self._target_reached = False
        self.min_vel = 0
        self.max_vel = 1
        self.min_omega = -math.pi
        self.max_omega = math.pi

    def target_reached(self, target, dist):
        distance_to_target = self.distance_to_entity(target)

        if distance_to_target < dist:
            self._target_reached = True

        return self._target_reached


class CuasEnvSingleAgent(BaseCuasEnv):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self, config=None):
        super().__init__()

        self.viewer = None

        # env is 80 meters x 60 meters
        self.env_width = 80
        self.env_height = 60

        self.screen_width = 800
        self.screen_height = 600

        self.num_attackers = 1
        self.num_agents = 1
        self.num_entities = self.num_attackers + self.num_agents
        self.time_elapse_penalty = 0
        self.prev_dist_to_target = 0
        self.prev_ang_to_target = 0

        self.max_omega = math.pi * 1.5  # rad/s
        self.min_omega = -self.max_omega
        self.max_vel = 2 * 1.5  # meters/s
        self.min_vel = 0
        self.max_distance = math.sqrt(self.env_width ** 2 + self.env_height ** 2)
        self.target_nuclear_rad = 10
        self.sensing_radius = 30
        self.assess_radius = 20
        self.target_width = 10
        self.target_height = 10

        # action is velocity and angular vel
        self.action_space = spaces.Box(
            low=np.array([self.min_vel, self.min_omega]),
            high=np.array([self.max_vel, self.max_omega]),
            dtype=np.float32,
            shape=(2,),
        )

        # x, y, theta, v, omega, dist_to_atacker, bearing_to_attacker, d_attacker_to_target, bearing_attacker_to_target
        self.observation_space = spaces.Box(
            low=np.array(
                [0, 0, -math.pi, self.min_vel, self.min_omega, 0, -math.pi, 0, -math.pi]
            ),
            high=np.array(
                [
                    self.env_width,
                    self.env_height,
                    math.pi,
                    self.max_vel,
                    self.max_omega,
                    self.max_distance,
                    math.pi,
                    self.max_distance,
                    math.pi,
                ]
            ),
            dtype=np.float32,
            shape=(9,),
        )
        self.reset()
        self.seed()

    def seed(self, seed=None):
        """ Random value to seed"""

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """resets the environment

        Arena is 80 meters by 60 meters

        Returns:
            [type]: [description]
        """
        self.defender_state = Agent(
            np.random.random() * 50,
            np.random.random() * 27,
            np.random.random() * math.pi,
        )
        # self.defender_state = Agent(68, 30, math.pi)

        # self.attacker_state = Attacker(75, 30, np.random.random()*math.pi)
        self.attacker_state = Attacker(
            50 + np.random.random() * 25,
            15 + np.random.random() * 15,
            np.random.random() * math.pi,
        )

        self.target_state = EntityState(40, 30, 0)
        self.target_state.width = self.target_width
        self.target_state.height = self.target_height

        self.time_elapse_penalty = 0
        self.prev_dist_to_attacker = self.defender_state.distance_to_entity(
            self.attacker_state
        )
        self.prev_ang_to_target = self.defender_state.bearing_to_entity(
            self.attacker_state
        )

        return self.get_observation()

    def attacker_move(
        self,
    ):
        kr = 6
        ka = 20

        alpha = 1
        dist_to_target = self.attacker_state.distance_to_entity(self.target_state)
        dist_to_defender = self.attacker_state.distance_to_entity(self.defender_state)

        # unit vector from attacker to defender and attacker to target
        des_vx = (
            -kr
            * (1 / (dist_to_defender) ** alpha)
            * ((self.defender_state.x - self.attacker_state.x) / dist_to_defender)
        )
        des_vx += (
            ka
            * (1 / (dist_to_target) ** alpha)
            * ((self.target_state.x - self.attacker_state.x) / dist_to_target)
        )

        des_vy = (
            -kr
            * (1 / (dist_to_defender) ** alpha)
            * ((self.defender_state.y - self.attacker_state.y) / dist_to_defender)
        )
        des_vy += (
            ka
            * (1 / (dist_to_target) ** alpha)
            * ((self.target_state.y - self.attacker_state.y) / dist_to_target)
        )

        des_v = np.array([des_vx, des_vy])

        dxu = self.si_uni_dyn(self.attacker_state, des_v)

        self.attacker_state.update_state(dxu[0], dxu[1])
        self.attacker_state = self.check_bounds(self.attacker_state)

    def si_uni_dyn(self, agent, si_v, projection_distance=0.05):
        cs = np.cos(agent.theta)
        ss = np.sin(agent.theta)

        dxu = np.zeros(
            2,
        )
        dxu[0] = cs * si_v[0] + ss * si_v[1]
        dxu[1] = (1 / projection_distance) * (-ss * si_v[0] + cs * si_v[1])

        dxu = np.clip(
            dxu, [agent.min_vel, agent.min_omega], [agent.max_vel, agent.max_omega]
        )

        return dxu

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        action = np.clip(action, [0, -self.max_omega], [self.max_vel, self.max_omega])
        # action = np.zeros(2,)

        # init rewards
        time_step_reward = -0.001
        target_reached_reward = 0
        attacker_neutralized_reward = 0
        reward_cordon = 0

        self.defender_state.update_state(action[0], action[1])
        self.defender_state = self.check_bounds(self.defender_state)

        self.attacker_move()

        self.time_elapse_penalty += time_step_reward

        attacker_dist_to_target = self.attacker_state.distance_to_entity(
            self.target_state
        )
        defender_dist_to_attacker = self.defender_state.distance_to_entity(
            self.attacker_state
        )
        defender_bearing_to_attacker = self.defender_state.bearing_to_entity(
            self.attacker_state
        )

        reward_change_in_dist = 0.9 * (
            self.prev_dist_to_attacker - defender_dist_to_attacker
        )
        reward_change_in_ang = 1 * (
            self.prev_ang_to_target - defender_bearing_to_attacker
        )

        self.prev_dist_to_attacker = defender_dist_to_attacker
        self.prev_ang_to_target = defender_bearing_to_attacker

        # this aids ddpg in training
        if attacker_dist_to_target < self.sensing_radius:
            reward_cordon = -0.05 * (
                1 - (attacker_dist_to_target / self.sensing_radius)
            )
        # reward_dist_to_attacker = .04 * (1 - (defender_dist_to_attacker / self.max_distance))

        attacker_neutralized = self.defender_state.attacker_neutralized(
            self.attacker_state
        )

        target_reached = self.attacker_state.target_reached(
            self.target_state, self.target_nuclear_rad
        )

        if attacker_neutralized:
            attacker_neutralized_reward += 10

        if target_reached:
            target_reached_reward = -10
        obs = self.get_observation()

        reward = (
            time_step_reward
            + target_reached_reward
            + attacker_neutralized_reward
            + reward_change_in_dist
            + reward_change_in_ang
            + reward_cordon
        )

        done = bool(
            target_reached or (self.time_elapse_penalty < -5) or attacker_neutralized
        )
        return (
            obs,
            reward,
            done,
            {
                "time_step_reward": time_step_reward,
                "target_reached_reward": target_reached_reward,
                "attacker_neutralized_reward": attacker_neutralized_reward,
                "reward_change_in_dist": reward_change_in_dist,
                "reward_change_in_ang": reward_change_in_ang,
                "reward_cordon": reward_cordon,
            },
        )

    def check_bounds(self, obj):
        """Checks bound for moving objects"""

        min_x = -obj.width / 2
        min_y = -obj.height / 2
        max_x = self.env_width + obj.width / 2
        max_y = self.env_height + obj.height / 2

        obj.x = min(max(0, obj.x), self.env_width)
        obj.y = min(max(0, obj.y), self.env_height)

        obj.theta = min(max(-math.pi, obj.theta), math.pi)

        return obj

    def get_observation(self):
        dist_to_attacker = self.defender_state.distance_to_entity(self.attacker_state)
        bearing_to_attacker = self.defender_state.bearing_to_entity(self.attacker_state)

        dist_attacker_to_target = self.attacker_state.distance_to_entity(
            self.target_state
        )
        bearing_attacker_to_target = self.attacker_state.bearing_to_entity(
            self.target_state
        )

        return np.squeeze(
            np.array(
                [
                    *self.defender_state.get_state(),
                    dist_to_attacker,
                    bearing_to_attacker,
                    dist_attacker_to_target,
                    bearing_attacker_to_target,
                ],
                dtype=np.float32,
            )
        )  # squeeze forces single dimension

    def render(self, mode="human"):

        x_scale = self.screen_width / self.env_width
        y_scale = self.screen_height / self.env_height

        if self.viewer is None:
            from cuas.envs import rendering

            protected_space = rendering.Image(
                fname=str(RESOURCE_FOLDER / "prot_space_blank.png"),
                width=self.target_state.width * x_scale,
                height=self.target_state.height * y_scale,
            )

            protected_space_trans = rendering.Transform(
                translation=(
                    self.target_state.x * x_scale,
                    self.target_state.y * y_scale,
                ),
                rotation=self.target_state.theta,
            )
            protected_space.add_attr(protected_space_trans)

            sensing_space = rendering.make_circle(
                self.sensing_radius * x_scale, filled=True
            )
            sensing_space.set_color(1, 1, 0)
            sensing_space.add_attr(protected_space_trans)

            assess_space = rendering.make_circle(
                self.assess_radius * x_scale, filled=False
            )
            assess_space.set_color(0, 1, 0)
            assess_space.add_attr(protected_space_trans)

            neutralize_space = rendering.make_circle(
                self.target_nuclear_rad * x_scale, filled=False
            )
            neutralize_space.set_color(1, 0, 0)
            neutralize_space.add_attr(protected_space_trans)

            self.defender = rendering.Image(
                fname=str(RESOURCE_FOLDER / "pursuer.png"),
                width=self.defender_state.width * x_scale,
                height=self.defender_state.height * y_scale,
            )

            self.defender_trans = rendering.Transform(
                translation=(
                    self.defender_state.x * x_scale,
                    self.defender_state.y * y_scale,
                ),
                rotation=self.defender_state.theta,
            )
            self.defender.add_attr(self.defender_trans)

            self.attacker = rendering.Image(
                fname=str(RESOURCE_FOLDER / "evader.png"),
                width=self.attacker_state.width * x_scale,
                height=self.attacker_state.height * y_scale,
            )

            self.attacker_trans = rendering.Transform(
                translation=(
                    self.attacker_state.x * x_scale,
                    self.attacker_state.y * y_scale,
                ),
                rotation=self.attacker_state.theta,
            )

            self.attacker.add_attr(self.attacker_trans)

            attacker_neutralize_rad = rendering.make_circle(
                self.attacker_state.neutralize_radius * x_scale, filled=False
            )
            attacker_neutralize_rad.set_color(1, 0, 0)
            attacker_neutralize_rad.add_attr(self.attacker_trans)

            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            self.viewer.add_geom(sensing_space)
            self.viewer.add_geom(protected_space)
            self.viewer.add_geom(self.defender)
            self.viewer.add_geom(self.attacker)
            self.viewer.add_geom(attacker_neutralize_rad)
            self.viewer.add_geom(assess_space)
            self.viewer.add_geom(neutralize_space)

        self.defender_trans.set_translation(
            self.defender_state.x * x_scale, self.defender_state.y * y_scale
        )
        self.defender_trans.set_rotation(self.defender_state.theta)

        self.attacker_trans.set_translation(
            self.attacker_state.x * x_scale, self.attacker_state.y * y_scale
        )
        self.attacker_trans.set_rotation(self.attacker_state.theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
