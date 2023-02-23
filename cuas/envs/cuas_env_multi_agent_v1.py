import os
import pathlib
import sys

import gym
import numpy as np
from cuas import util
from cuas.agents.cuas_agents import (
    Agent,
    AgentType,
    CuasAgent,
    CuasAgent2D,
    Entity,
    Obstacle,
    ObsType,
)
from cuas.envs.base_cuas_env import BaseCuasEnv
from cuas.policy.velocity_obstacle import compute_velocity
from cuas.safety_layer.safety_layer import SafetyLayer
from gym import spaces
from gym.error import DependencyNotInstalled
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.spaces.repeated import Repeated
from qpsolvers import solve_qp

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
RESOURCE_FOLDER = path.joinpath("../../resources")


metadata = {
    "render_modes": ["human", "rgb_array", "single_rgb_array"],
    "render_fps": 30,
}


class CuasEnvMultiAgentV1(BaseCuasEnv, MultiAgentEnv):
    def __init__(self, env_config={}):
        super().__init__()
        self.viewer = None

        self.config = env_config
        self._parse_config()
        self.seed(seed=self.sim_seed)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self._past_obs = self.reset()
        self.num_constraints = 1 + (self.num_pursuers - 1) + self.num_obstacles
        self.safety_layer = None

    def _parse_config(self):
        self.env_width = self.config.get("env_width", 80)
        self.env_height = self.config.get("env_height", 60)
        self.screen_width = self.config.get("screen_width", 800)
        self.screen_height = self.config.get("screen_height", 600)

        # setting number of agents
        self.max_num_pursuers = self.config.get("max_num_pursuers", 10)
        self.num_pursuers = self.config.get("num_pursuers", 3)
        # agents is synonymous with pursuers
        self.num_agents = self.num_pursuers
        self.max_num_agents = self.max_num_pursuers
        assert self.num_pursuers <= self.max_num_pursuers, print(
            f"Number of agents {self.num_pursuers} must be less than Max number of agents {self.max_num_pursuers}"
        )

        # setting number of evaders
        self.max_num_evaders = self.config.get("max_num_evaders", 10)
        self.num_evaders = self.config.get("num_evaders", 1)
        assert self.num_evaders <= self.max_num_evaders, print(
            f"Number of evaders {self.num_evaders} must be less than MAX number of evaders {self.max_num_evaders}"
        )

        # setting number of obstacles
        self.max_num_obstacles = self.config.get("max_num_obstacles", 10)
        self.num_obstacles = self.config.get("num_obstacles", 0)
        assert self.num_obstacles <= self.max_num_obstacles, print(
            f"Number of obstacles {self.num_obstacles} must be less than MAX number of obstacles {self.max_num_obstacles}"
        )

        self._constraint_slack = self.config.get("constraint_slack", 1.5606)
        self._constraint_k = self.config.get("constraint_k", 1)
        self.sim_seed = self.config.get("seed", None)
        self.time_step_penalty = self.config.get("time_step_penalty", 0)
        self.norm_env = self.config.get("normalize", True)
        self.prior_policy_mixin = self.config.get("prior_policy_mixin", 0)
        self.max_distance = util.distance(point_2=(self.env_width, self.env_height))

        self.pursuer_drive_type = getattr(
            sys.modules[__name__], self.config.get("pursuer_drive_type", "CuasAgent")
        )

        self.agent_radius = self.config.get("agent_radius", 1)
        self.agent_v_min = self.config.get("v_min", 0)
        self.agent_v_max = self.config.get("v_max", 10)  # m/s
        self.agent_w_min = self.config.get("w_min", -np.pi * 2)  # rad/s
        self.agent_w_max = self.config.get("w_max", np.pi * 2)
        # self.agent_w_min = self.config.get("w_min", -np.pi * 2)  # rad/s
        # self.agent_w_max = self.config.get("w_max", np.pi * 2)
        self.observation_radius = self.config.get("observation_radius", 20)
        self.show_observation_radius = self.config.get("show_observation_radius", True)
        # angle convert radians. It's divided by 2 because FOv is to left and right of heading
        self.observation_fov = (
            self.config.get("observation_fov", 30) * util.DEG2RAD
        )  # angle convert to radians.

        # options are local or global
        self.observation_type = self.config.get("observation_type", "global")

        # when observation type is global, we set the observation radius to max size of environment
        if self.observation_type == "global":
            self.observation_radius = self.max_distance
            print(
                "Setting show_observation_radius to False since observation_type  is global."
            )
            self.show_observation_radius = False
        else:
            # observation radius should be less than max_distance
            assert (
                self.observation_radius <= self.max_distance
            ), f"Observation radius {self.observation_radius} must be less than environment size {self.max_distance}"

        self.time_step = self.config.get("time_step", 0.01)
        self.max_time = self.config.get("max_time", 40)
        self.alpha = self.config.get("alpha", 0.0)
        self.beta = self.config.get("beta", 0.0025)
        self.evader_move_type = self.config.get("evader_move_type", "repulsive")
        self.pursuer_move_type = self.config.get("pursuer_move_type", "rl")
        self.evader_alpha = self.config.get("evader_alpha", 1)
        self.obstacle_radius = self.config.get("obstacle_radius", 1)
        self.obstacle_v = self.config.get("obstacle_v", 1)
        self.obstacle_penalty = self.config.get("obstacle_penalty", False)
        self.agent_collision_penalty = self.config.get("agent_collision_penalty", False)
        self.agent_penalty_weight = self.config.get("agent_penalty_weight", 0.25)
        self.obstacle_penalty_weight = self.config.get("obstacle_penalty_weight", 0.25)
        self.render_trace = self.config.get("render_trace", False)

        self.norm_low = np.array([-1.0, -1.0])
        self.norm_high = np.array([1.0, 1.0])

        self.low = np.array([self.agent_v_min, self.agent_w_min])
        self.high = np.array([self.agent_v_max, self.agent_w_max])

        self.target = Entity(
            x=self.config.get("target_x", 40),
            y=self.config.get("target_y", 30),
            r=self.config.get("target_radius", 5),
        )

        self.target.alarmed_radius = self.config.get("alarmed_radius", 30)
        self.use_safety_layer = self.config.get("use_safety_layer", False)
        self.safety_layer_type = self.config.get("safety_layer_type", "soft")
        self.use_safe_action = self.config.get("use_safe_action", False)

    def load_safety_layer(self):
        print("using safety layer")
        self.safety_layer = SafetyLayer(self)
        self.safety_layer.load_layer(self.config.get("sl_checkpoint_dir", None))
        self.corrective_action = self.safety_layer.get_safe_actions
        # if self.safety_layer_type == "hard":
        #     self.corrective_action = self.safety_layer.get_hard_safe_action
        # elif self.safety_layer_type == "soft":
        #     self.corrective_action = self.safety_layer.get_soft_safe_action
        # else:
        #     raise ValueError("Unknown safety layer type")

    def _add_obstacles(self):
        def get_random_obstacle():
            x = np.random.random() * self.env_width
            y = np.random.random() * self.env_height
            theta = np.random.random() * np.pi
            # r = np.random.random() * self.obstacle_radius + 1
            r = self.obstacle_radius
            # obs_type = np.random.randint(2)
            obs_type = ObsType.M

            return Obstacle(x, y, theta=theta, r=r, obs_type=obs_type)

        for _ in range(self.num_obstacles):
            temp_obstacle = get_random_obstacle()
            temp_obstacle.v = self.obstacle_v
            self.obstacles.append(temp_obstacle)

    def _create_agents(self, agent_type, num_pursuers):
        def is_collision(agent):
            # evaders can't start in arena close to target
            if agent.type == AgentType.E:
                dist_target = agent.rel_dist(self.target)
                if dist_target < self.target.alarmed_radius:
                    return True

            elif agent.type == AgentType.P:
                dist_target = agent.rel_dist(self.target)
                if dist_target > self.target.alarmed_radius:
                    return True

            if agent.collision_with(self.target):
                return True

            for obs in self.obstacles:
                if agent.collision_with(obs):
                    return True

            for _agent in self.agents:
                if agent.collision_with(_agent):
                    return True

            for evader in self.evaders:
                if agent.collision_with(evader):
                    return True

            return False

        def get_random_agent(agent_id):
            x = np.random.random() * self.env_width
            y = np.random.random() * self.env_height
            theta = np.random.random() * np.pi
            r = self.agent_radius

            if agent_type == AgentType.P:
                move_type = self.pursuer_move_type
                return self.pursuer_drive_type(
                    agent_id,
                    agent_type,
                    x,
                    y,
                    theta,
                    r=r,
                    obs_r=self.observation_radius,
                    move_type=move_type,
                )
            elif agent_type == AgentType.E:
                move_type = self.evader_move_type
                return CuasAgent(
                    agent_id,
                    agent_type,
                    x,
                    y,
                    theta,
                    r=r,
                    obs_r=self.observation_radius,
                    move_type=move_type,
                )
            else:
                raise TypeError("Unknown Agent Type")

        for agent_id in range(num_pursuers):
            in_collision = True
            # make sure agents start in correct positions
            while in_collision:
                agent = get_random_agent(agent_id)
                in_collision = is_collision(agent)

            if agent_type == AgentType.P:
                self.agents.append(agent)
            else:
                self.evaders.append(agent)

    def _get_observation_space(self):
        """
        Returns a repeated space for each agent. See: https://github.com/ray-project/ray/blob/master/rllib/examples/env/simple_rpg.py
        """
        # self.agent_num_states = 7
        # obs_num_state = 2
        # evader_other_agent_num_state = 5

        # # agent_state: x, y, theta, v, w, dist_target, theta_target
        # agent_state = spaces.Box(
        #     low=np.array([-1] * self.agent_num_states),
        #     high=np.array([1] * self.agent_num_states),
        #     dtype=np.float32,
        # )

        # # evader_state: dist_evader, theta_evader, evader_rel_bearing, evader_dist_target, evader_rel_bearing_target
        # evader_state = spaces.Box(
        #     low=np.array([-1] * evader_other_agent_num_state),
        #     high=np.array([1] * evader_other_agent_num_state),
        #     dtype=np.float32,
        # )

        # # other_agents: dist_agent_j, theta_agent_j, agent_j_rel_bearing, agent_j_dist_target, agent_j_rel_bearing_target
        # other_agents_state = spaces.Box(
        #     low=np.array([-1] * evader_other_agent_num_state),
        #     high=np.array([1] * evader_other_agent_num_state),
        #     dtype=np.float32,
        # )

        # # obstacle: distance_obstacle_k, theta_obstacle_k
        # obs_state = spaces.Box(
        #     low=np.array([-1] * obs_num_state),
        #     high=np.array([1] * obs_num_state),
        #     dtype=np.float32,
        # )

        # self.agent_low_obs = np.array(
        #     [
        #         0,
        #         0,
        #         -np.pi,
        #         self.agent_v_min,
        #         self.agent_w_min,
        #         0,
        #         -np.pi,
        #     ]
        # )

        # self.agent_high_obs = np.array(
        #     [
        #         self.env_width,
        #         self.env_height,
        #         np.pi,
        #         self.agent_v_max,
        #         self.agent_w_max,
        #         self.max_distance,
        #         np.pi,
        #     ]
        # )

        # self.evader_low_obs = np.array([0, -np.pi, -np.pi, 0, -np.pi])

        # self.evader_high_obs = np.array(
        #     [
        #         self.max_distance,
        #         np.pi,
        #         np.pi,
        #         self.max_distance,
        #         np.pi,
        #     ]
        # )

        # self.other_agents_low_obs = np.array([[0, -np.pi, -np.pi, 0, -np.pi]])
        # self.other_agents_high_obs = np.array(
        #     [
        #         self.max_distance,
        #         np.pi,
        #         np.pi,
        #         self.max_distance,
        #         np.pi,
        #     ]
        # )

        # self.obstacles_low_obs = np.array([0, -np.pi])

        # self.obstacles_high_obs = np.array([self.max_distance, np.pi])

        # self.agent_space = spaces.Dict(
        #     {
        #         "agent_state": agent_state,
        #         "evader_state": Repeated(evader_state, max_len=self.max_num_evaders),
        #         "other_agent_state": Repeated(
        #             other_agents_state, max_len=self.max_num_pursuers - 1
        #         ),
        #         "obs_state": Repeated(obs_state, max_len=self.max_num_obstacles),
        #     }
        # )

        self.agent_num_states = 7
        self.obs_num_states = 2
        self.evader_other_agent_num_states = 5
        # agent_state: x, y, theta, v, w, dist_target, theta_target
        self.norm_low_obs = [-1] * self.agent_num_states
        # evader_state: type, sensed, dist_evader, theta_evader, evader_rel_bearing, evader_dist_target, evader_rel_bearing_target
        self.norm_low_obs.extend(
            ([-1] * self.evader_other_agent_num_states) * (self.max_num_evaders)
        )
        # other_agents: type, sensed, dist_agent_j, theta_agent_j, agent_j_rel_bearing, agent_j_dist_target, agent_j_rel_bearing_target
        # subtract 1 to not include the current agent
        self.norm_low_obs.extend(
            ([-1] * self.evader_other_agent_num_states) * (self.max_num_pursuers - 1)
        )
        # obstacle: type, sensed, distance_obstacle_k, theta_obstacle_k
        self.norm_low_obs.extend(
            ([-1] * self.obs_num_states) * (self.max_num_obstacles)
        )
        self.norm_low_obs = np.array(self.norm_low_obs)
        self.norm_high_obs = np.array([1] * self.norm_low_obs.size)

        # the main agent
        self.low_obs = [
            0,
            0,
            -np.pi,
            self.agent_v_min,
            self.agent_w_min,
            0,
            -np.pi,
        ]

        # evader states
        # type, sensed, rel_dist, rel_bearing, rel_bearing_self, rel_dist_target, rel_bearing_target
        self.low_obs.extend([0, -np.pi, -np.pi, 0, -np.pi] * (self.max_num_evaders))

        # other agent states
        # type, sensed, rel_dist, rel_bearing, rel_bearing_self, rel_dist_target, rel_bearing_target
        self.low_obs.extend(
            [0, -np.pi, -np.pi, 0, -np.pi] * (self.max_num_pursuers - 1)
        )

        # obstacle states
        # type, sensed, rel_dist, rel_bearing
        self.low_obs.extend([0, -np.pi] * (self.max_num_obstacles))

        self.low_obs = np.array(self.low_obs)

        # the main agent
        self.high_obs = [
            self.env_width,
            self.env_height,
            np.pi,
            self.agent_v_max,
            self.agent_w_max,
            self.max_distance,
            np.pi,
        ]

        # evader states
        # type, sensed, rel_dist, rel_bearing, rel_bearing_self, rel_dist_target, rel_bearing_target
        self.high_obs.extend(
            [
                # len(AgentType),
                # 1,
                self.max_distance,
                np.pi,
                np.pi,
                self.max_distance,
                np.pi,
            ]
            * (self.max_num_evaders)
        )
        # other_agent states
        # type, sensed, rel_dist, rel_bearing, rel_bearing_self, rel_dist_target, rel_bearing_target
        self.high_obs.extend(
            [
                # len(AgentType),
                # 1,
                self.max_distance,
                np.pi,
                np.pi,
                self.max_distance,
                np.pi,
            ]
            * (self.max_num_pursuers - 1)
        )

        # obstacles
        # type, sensed
        self.high_obs.extend(
            [
                # len(AgentType), 1,
                self.max_distance,
                np.pi,
            ]
            * (self.max_num_obstacles)
        )

        self.high_obs = np.array(self.high_obs)

        return spaces.Dict(
            {
                i: spaces.Dict(
                    {
                        "observations": spaces.Box(
                            low=self.norm_low_obs,
                            high=self.norm_high_obs,
                            dtype=np.float32,
                        ),
                        "raw_obs": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            dtype=np.float32,
                            shape=(self.norm_low_obs.shape[0],),
                        ),
                        "constraints": spaces.Box(
                            low=-np.inf,
                            high=np.inf,
                            shape=(1 + self.num_pursuers - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        "action_g": spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(1 + self.num_pursuers - 1 + self.num_obstacles, 2),
                            dtype=np.float32,
                        ),
                        "action_h": spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(1 + self.num_pursuers - 1 + self.num_obstacles,),
                            dtype=np.float32,
                        ),
                        "action_r": spaces.Box(
                            -1.0,
                            1.0,
                            shape=(2, 2),
                            dtype=np.float32,
                        ),
                        #     "action_a": spaces.Box(
                        #         low=self.norm_low, high=self.norm_high, dtype=np.float32
                        #     ),
                    }
                )
                for i in range(self.num_pursuers)
            }
        )

    def _get_action_space(self):
        """
        The number of action will be a fix size
        """
        return spaces.Dict(
            {
                i: spaces.Box(low=self.norm_low, high=self.norm_high, dtype=np.float32)
                for i in range(self.num_pursuers)
            }
        )

    def reset(self):
        """
        Reset environment
        """
        if self.viewer is not None:
            self.close()

        self.dones = set()
        self.obstacles = []
        self._add_obstacles()
        self.agents = []
        self.evaders = []
        self._create_agents(AgentType.P, self.num_pursuers)
        self._create_agents(AgentType.E, self.num_evaders)
        self.cum_a_bar = {agent.id: np.array([0.0, 0.0]) for agent in self.agents}
        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}
        self.time_elapse = 0
        self.target_done = False
        self._agent_ids = {agent.id for agent in self.agents}

        return obs

    def seed(self, seed=None):
        """Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def repulsive_action(self, agent):
        """https://link.springer.com/article/10.1007/s10514-020-09945-6"""
        # kr = 10
        # # ka should be 10
        # ka = 10 * kr
        kr = 220
        ka = 200

        target = self.evaders[0] if agent.type == AgentType.P else self.target

        dist_to_target = agent.rel_dist(target) + 0.001
        agent_q = np.array([agent.x, agent.y])
        target_q = np.array([target.x, target.y])

        target_q_star = 1 * (target.radius + agent.radius)
        if dist_to_target <= target_q_star:
            des_v = -ka * (agent_q - target_q)
        else:
            des_v = (
                -ka
                * (1 / dist_to_target**self.evader_alpha)
                * ((agent_q - target_q) / dist_to_target)
            )

        # pursuer agents potential
        # only use when close to obstacle
        for other_agent in self.agents:
            if agent.type == AgentType.P and agent.id == other_agent.id:
                continue
            other_agent_q_star = 5 * agent.radius

            dist_to_other_agent = agent.rel_dist(other_agent)
            other_agent_q = np.array([other_agent.x, other_agent.y])

            if dist_to_other_agent <= other_agent_q_star:
                des_v += (
                    kr
                    * ((1 / dist_to_other_agent) - (1 / other_agent_q_star))
                    * (1 / dist_to_other_agent**self.evader_alpha)
                    * ((agent_q - other_agent_q) / dist_to_other_agent)
                )

        if agent.type == AgentType.E:
            for other_evader in self.evaders:
                other_evader_q_star = 5 * agent.radius
                if other_evader.id == agent.id:
                    continue

                dist_to_other_evader = agent.rel_dist(other_evader)
                other_evader_q = np.array([other_evader.x, other_evader.y])

                if dist_to_other_evader <= other_evader_q_star:
                    des_v += (
                        kr
                        * ((1 / dist_to_other_evader) - (1 / other_evader_q_star))
                        * (1 / dist_to_other_evader**self.evader_alpha)
                        * ((agent_q - other_evader_q) / dist_to_other_evader)
                    )

        for obstacle in self.obstacles:
            dist_to_obstacle = agent.rel_dist(obstacle) + 0.001
            obstacle_q = np.array([obstacle.x, obstacle.y])
            obstacle_q_star = 5 * (agent.radius + obstacle.radius)

            if dist_to_obstacle <= obstacle_q_star:
                des_v += (
                    kr
                    * ((1 / dist_to_obstacle) - (1 / obstacle_q_star))
                    * (1 / dist_to_obstacle**self.evader_alpha)
                    * ((agent_q - obstacle_q) / dist_to_obstacle)
                )
        # if dist_to_target >= 0.05:

        #     # des_vx = (
        #     #     ka
        #     #     * (1 / (dist_to_target ** self.evader_alpha))
        #     #     * ((target.x - agent.x) / dist_to_target)
        #     # )
        #     # des_vy = (
        #     #     ka
        #     #     * (1 / (dist_to_target ** self.evader_alpha))
        #     #     * ((target.y - agent.y) / dist_to_target)
        #     # )
        #     des_vx = ka * (target.x - agent.x)
        #     des_vy = ka * (target.y - agent.y)

        #     rep_vx = 0
        #     rep_vy = 0
        #     for other_agent in self.agents:
        #         if other_agent.id == agent.id:
        #             continue

        #         dist_to_other_agent = agent.rel_dist(other_agent) + epsilon

        #         rep_vx += (1 / (dist_to_other_agent ** self.evader_alpha)) * (
        #             (other_agent.x - agent.x) / dist_to_other_agent
        #         )

        #         rep_vy += (1 / (dist_to_other_agent ** self.evader_alpha)) * (
        #             (other_agent.y - agent.y) / dist_to_other_agent
        #         )

        #     for obs in self.obstacles:
        #         dist_to_obstacle = agent.rel_dist(obs) + epsilon

        #         rep_vx += (1 / (dist_to_obstacle ** self.evader_alpha)) * (
        #             (obs.x - agent.x) / dist_to_obstacle
        #         )

        #         rep_vy += (1 / (dist_to_obstacle ** self.evader_alpha)) * (
        #             (obs.y - agent.y) / dist_to_obstacle
        #         )

        #     if agent.type == AgentType.P:
        #         dist_to_target = agent.rel_dist(self.target) + epsilon

        #         rep_vx += (1 / (dist_to_target ** self.evader_alpha)) * (
        #             (self.target.x - agent.x) / dist_to_target
        #         )

        #         rep_vy += (1 / (dist_to_target ** self.evader_alpha)) * (
        #             (self.target.y - agent.y) / dist_to_target
        #         )

        #     des_vx += -kr * rep_vx
        #     des_vy += -kr * rep_vy
        #     des_v = (
        #         self.agent_v_max
        #         * np.array([des_vx, des_vy])
        #         / np.linalg.norm([des_vx, des_vy])
        #     )

        #     dxu = self.si_uni_dyn(agent, des_v)
        #     # dx_dy = des_v * .01

        #     # des_theta = np.arctan2(dx_dy[1], dx_dy[0])
        #     # omega = 10 * (agent.theta - des_theta)
        #     # v_norm = np.linalg.norm(des_v)
        #     # # dxu =np.array( [v_norm, omega])
        # else:
        #     dxu = np.array([0, 0])
        des_v = self.agent_v_max * des_v
        dxu = self.si_uni_dyn(agent, des_v)

        return dxu

    # TODO: fix this
    def go_to_goal(self, agent):
        """Policy for Evader to move to goal"""
        # https://asl.ethz.ch/education/lectures/autonomous_mobile_robots/spring-2020.html
        k_rho = 0.5
        k_alpha = 2
        k_beta = -0.01
        target = self.target if agent.type == AgentType.E else self.evaders[0]
        rho = agent.rel_dist(target)
        alpha = -agent.theta + agent.rel_bearing(target)
        beta = -agent.theta - alpha
        vw = np.array([k_rho * rho, k_alpha * alpha + k_beta * beta])
        vw = np.clip(
            vw,
            [self.agent_v_min, self.agent_w_min],
            [self.agent_v_max, self.agent_w_max],
        )

        return vw

    @staticmethod
    def uni_to_si_dyn(agent, dxu, projection_distance=0.05):
        """
        See:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py

        """
        cs = np.cos(agent.theta)
        ss = np.sin(agent.theta)

        dxi = np.zeros(
            2,
        )
        dxi[0] = cs * dxu[0] - projection_distance * ss * dxu[1]
        dxi[1] = ss * dxu[0] + projection_distance * cs * dxu[1]

        return dxi

    # TODO: projection_distance=.01
    def si_uni_dyn(self, agent, si_v, projection_distance=0.05):
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
        cs = np.cos(agent.theta)
        ss = np.sin(agent.theta)

        dxu = np.zeros(
            2,
        )
        dxu[0] = cs * si_v[0] + ss * si_v[1]
        dxu[1] = (1 / projection_distance) * (-ss * si_v[0] + cs * si_v[1])

        dxu = np.clip(
            dxu,
            [self.agent_v_min, self.agent_w_min],
            [self.agent_v_max, self.agent_w_max],
        )

        return dxu

    def get_rl(self, theta, projection_distance=0.05):
        cs = np.cos(theta)
        ss = np.sin(theta)
        rl_array = np.array(
            [[cs, -projection_distance * ss], [ss, projection_distance * cs]]
        )

        return rl_array

    def _unscale_obs(self, obs):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        # print("action: ", action)
        # print("action type:", type(action))
        # unnormalize the action
        assert np.all(np.greater_equal(obs, self.norm_low_obs)), (
            obs,
            self.norm_low_obs,
        )
        assert np.all(np.less_equal(obs, self.norm_high_obs)), (obs, self.norm_high_obs)
        obs = self.low_obs + (self.high_obs - self.low_obs) * (
            (obs - self.norm_low_obs) / (self.norm_high_obs - self.norm_low_obs)
        )
        # obs = np.clip(action, self.low, self.high)

        return obs

    def _unscale_action(self, action):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        assert np.all(np.greater_equal(action, self.norm_low)), (action, self.norm_low)
        assert np.all(np.less_equal(action, self.norm_high)), (action, self.norm_high)
        action = self.low + (self.high - self.low) * (
            (action - self.norm_low) / (self.norm_high - self.norm_low)
        )
        # TODO: this is not needed
        action = np.clip(action, self.low, self.high)

        return action

    def calc_projected_states(self, agent):

        A = np.eye(2)
        G = []
        h = []

        # target
        G.append(-np.dot(2 * (agent.pos - self.target.pos), A))
        h.append(
            self._constraint_k
            * (
                np.linalg.norm(agent.pos - self.target.pos) ** 2
                - (agent.radius + self.target.radius + self._constraint_slack) ** 2
            )
        )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id:
                # z_other_agent = np.array([other_agent.x, other_agent.y])
                G.append(-np.dot(2 * (agent.pos - other_agent.pos), A))
                h.append(
                    self._constraint_k
                    * (
                        np.linalg.norm(agent.pos - other_agent.pos) ** 2
                        - (agent.radius + other_agent.radius + self._constraint_slack)
                        ** 2
                    )
                )

        # obstacles
        for obstacle in self.obstacles:
            # z_obstacle = np.array([obstacle.x, obstacle.y])
            G.append(-np.dot(2 * (agent.pos - obstacle.pos), A))
            h.append(
                self._constraint_k
                * (
                    np.linalg.norm(agent.pos - obstacle.pos) ** 2
                    - (agent.radius + obstacle.radius + self._constraint_slack) ** 2
                )
            )

        G = np.array(G)
        h = np.array(h)

        return G, h, self.get_rl(agent.theta)

    def safety_action_layer(self, agent, a_uni_rl):
        a_si_rl = agent.uni_to_si_dyn(a_uni_rl)

        A = np.dot(np.array([1, 1]), np.eye(2))
        P = np.eye(2)
        q = -np.dot(P.T, a_si_rl)

        G = []
        h = []

        if agent.sensed(self.target):
            G.append(A)
            h.append(
                -(agent.radius + self.target.radius + self._constraint_slack)
                + np.linalg.norm(agent.pos - self.target.pos)
            )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id and agent.sensed(other_agent):
                G.append(A)
                h.append(
                    -(agent.radius + other_agent.radius + self._constraint_slack)
                    + np.linalg.norm(agent.pos - other_agent.pos)
                )

        # obstacles
        for obstacle in self.obstacles:
            if agent.sensed(obstacle):
                G.append(A)
                h.append(
                    -(agent.radius + obstacle.radius + self._constraint_slack)
                    + np.linalg.norm(agent.pos - obstacle.pos)
                )
        G = np.array(G)
        h = np.array(h)
        if G.any() and h.any():
            try:
                a_si_qp = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                # just return 0 if infeasible action
                return a_uni_rl
        else:
            return a_uni_rl
            # a_si_qp = q

        # just return 0
        if a_si_qp is None:
            print("infeasible solver")
            return a_uni_rl
            # a_si_qp = q

        # convert to unicycle
        a_uni_qp = agent.si_to_uni_dyn(a_si_qp)
        # a_uni_qp = a_si_qp

        # a_uni_qp = np.clip(a_uni_rl + a_uni_qp, self.low, self.high)

        return a_uni_qp

    def proj_safe_action(self, agent, a_uni_rl):
        """_summary_

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        # a_si_rl = a_uni_rl
        a_si_rl = agent.uni_to_si_dyn(a_uni_rl)

        # A = np.dot(np.eye(2), self.get_rl(agent.theta))
        A = np.eye(2)

        P = np.eye(2)
        q = np.zeros(2)
        G = []
        h = []

        # target
        if agent.sensed(self.target):
            G.append(-np.dot(2 * (agent.pos - self.target.pos), A))
            h.append(
                self._constraint_k
                * (
                    np.linalg.norm(agent.pos - self.target.pos) ** 2
                    - (agent.radius + self.target.radius + self._constraint_slack) ** 2
                )
                + np.dot(2 * (agent.pos - self.target.pos), np.dot(A, a_si_rl))
            )

        # other agents
        for other_agent in self.agents:
            if agent.id != other_agent.id and agent.sensed(other_agent):
                # z_other_agent = np.array([other_agent.x, other_agent.y])
                G.append(-np.dot(2 * (agent.pos - other_agent.pos), A))
                h.append(
                    self._constraint_k
                    * (
                        np.linalg.norm(agent.pos - other_agent.pos) ** 2
                        - (agent.radius + other_agent.radius + self._constraint_slack)
                        ** 2
                    )
                    + np.dot(2 * (agent.pos - other_agent.pos), np.dot(A, a_si_rl))
                )

        # obstacles
        for obstacle in self.obstacles:
            # z_obstacle = np.array([obstacle.x, obstacle.y])
            if agent.sensed(obstacle):
                G.append(-np.dot(2 * (agent.pos - obstacle.pos), A))
                h.append(
                    self._constraint_k
                    * (
                        np.linalg.norm(agent.pos - obstacle.pos) ** 2
                        - (agent.radius + obstacle.radius + self._constraint_slack) ** 2
                    )
                    + np.dot(2 * (agent.pos - obstacle.pos), np.dot(A, a_si_rl))
                )

        G = np.array(G)
        h = np.array(h)

        if G.any() and h.any():
            try:
                a_si_qp = solve_qp(
                    P.astype(np.float64),
                    q.astype(np.float64),
                    G.astype(np.float64),
                    h.astype(np.float64),
                    None,
                    None,
                    None,
                    None,
                    solver="quadprog",
                )
            except Exception as e:
                print(f"error running solver: {e}")
                # just return 0 if infeasible action
                a_si_qp = q
        else:
            return q
            # a_si_qp = q

        # just return 0
        if a_si_qp is None:
            print("infeasible solver")
            return q
            # a_si_qp = q

        # convert to unicycle
        a_uni_qp = agent.si_to_uni_dyn(a_si_qp)
        # a_uni_qp = a_si_qp

        # a_uni_qp = np.clip(a_uni_rl + a_uni_qp, self.low, self.high)

        return a_uni_qp

    # TODO: caught agents shouldn't move
    def step(self, actions):
        """[summary]

        action is of type dictionary
        Args:
            action ([type], optional): [description]. Defaults to None.
        """

        self.time_elapse += self.time_step
        obs, rew, done, info = {}, {}, {}, {}

        if self.use_safety_layer:
            if self.safety_layer is None:
                self.load_safety_layer()

            actions = self.corrective_action(
                # {agent.id: self._calc_obs(agent) for agent in self.agents},
                self._past_obs,
                actions,
                # self.get_constraints(),
            )
            for i, a in actions.items():
                actions[i] = self._scale_action(a)
                actions[i] = np.clip(a, self.norm_low, self.norm_high)

        # agents move
        for i, action in actions.items():
            action = self._unscale_action(action)

            if self.agents[i].move_type == "go_to_goal":
                action = self.go_to_goal(self.agents[i])
            elif self.agents[i].move_type == "repulsive":
                action = self.repulsive_action(self.agents[i])
            # k_rho = 1
            # k_alpha = 3
            # k_beta = -0.5
            # rho = self.agents[i].rel_dist(self.evaders[0])
            # alpha = -self.agents[i].theta + self.agents[i].rel_bearing(self.evaders[0])
            # beta = -self.agents[i].theta - alpha
            # pro_action = np.array([k_rho * rho, k_alpha * alpha + k_beta * beta])

            # action = (1 / (1 + self.prior_policy_mixin)) * action + (
            #     self.prior_policy_mixin / (1 + self.prior_policy_mixin)
            # ) * pro_action
            # action = np.clip(action, self.low, self.high)
            if self.use_safe_action:
                action_qp = self.proj_safe_action(self.agents[i], action)
                self.cum_a_bar[i] = np.clip(
                    self.cum_a_bar[i] + action_qp, self.low, self.high
                )
                action += action_qp

                # temp safety_action_layer
                # action = self.safety_action_layer(self.agents[i], action)
                action = np.clip(action, self.low, self.high)
            self.agents[i].step(action)

        # evader moves
        for evader in self.evaders:
            if evader.done:
                evader.move_type = "static"
            self._agent_step(evader, None)

        # obstacle moves
        for obstacle in self.obstacles:
            obstacle.step()

        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}
        rew = {agent.id: self._calc_reward(agent) for agent in self.agents}
        done = self._get_done()
        info = self._calc_info()

        self._past_obs = obs

        return obs, rew, done, info

    def _agent_step(self, agent, action=None, unscale_action=True):
        """
        Handles how any agent moves in the environment. Agent can be pursuer or evader.

        Args:
            agent (_type_): _description_
            action (_type_, optional): _description_. Defaults to None.
        """
        if agent.done or agent.move_type == "static":
            action = np.array([0, 0])

        elif agent.move_type == "rl":
            # action = self._unscale_action(action)
            pass

        elif agent.move_type == "repulsive":
            unscale_action = False
            action = self.repulsive_action(agent)

        elif agent.move_type == "random":
            action = self.action_space[0].sample()

        elif agent.move_type == "go_to_goal":
            unscale_action = False
            action = self.go_to_goal(agent)

        # if self.use_safety_layer and agent.type == AgentType.P:
        #     if self.safety_layer is None:
        #         self.load_safety_layer()

        #     action = self.safety_layer.get_safe_action(
        #         self._calc_obs(agent), action, self._get_agent_constraint(agent)
        #     )

        if unscale_action:
            action = self._unscale_action(action)
        agent.step(action)

    def _calc_info(self):
        """Provides info to calculate metric for scenario

        Returns:
            _type_: _description_
        """
        info = {}
        for agent in self.agents:
            target_collision = 0
            obstacle_collision = 0
            agent_collision = 0

            # there can only be one target collision
            target_collision = 1 if agent.collision_with(self.target) else 0

            for other_agent in self.agents:
                if agent.id == other_agent.id:
                    continue
                agent_collision += 1 if agent.collision_with(other_agent) else 0

            for obstacle in self.obstacles:
                obstacle_collision += 1 if agent.collision_with(obstacle) else 0

            # TODO: need to find a way to reset this so that we only get the number of evader captured per round
            evader_captured = 0
            target_breached = 0
            for evader in self.evaders:
                evader_captured += 1 if evader.captured else 0
                target_breached += 1 if evader.collision_with(self.target) else 0

            info[agent.id] = {
                "target_collision": target_collision,
                "agent_collision": agent_collision,
                "obstacle_collision": obstacle_collision,
                "evader_captured": evader_captured,
                "target_breached": target_breached,
                "agent_action": self._scale_action(np.array([agent.v, agent.w])),
            }

        return info

    def _calc_reward(self, agent):
        """Calculate rewards for each agent

        Args:
            agent (_type_): _description_

        Returns:
            _type_: _description_
        """
        reward = 0

        evader = self.evaders[0]

        if not evader.done:
            if agent.collision_with(evader):
                evader.done = True
                evader.captured = True
                reward += 1

            # penalty for evader reaching target
            elif evader.collision_with(self.target):
                self.target_done = True
                evader.done = True
                reward += -1

            # this only get calculated if the evader is not caught or didn't reach the target
            elif agent.sensed(evader):
                dist_evader = agent.rel_dist(evader)
                reward += -self.beta * (dist_evader / self.observation_radius)

            else:
                reward += -self.beta

        # evader_dist = []
        # for evader in self.evaders:
        #     if not evader.done:
        #         if agent.collision_with(evader):
        #             evader.done = True
        #             evader.captured = True
        #             reward += 1
        #             break

        #         # penalty for evader reaching target
        #         if evader.collision_with(self.target):
        #             self.target_done = True
        #             evader.done = True
        #             reward += -1
        #             break

        #         # this only get calculated if the evader is not caught or didn't reach the target
        #         if agent.sensed(evader):
        #             evader_dist.append(agent.rel_dist(evader))

        # if evader_dist:
        #     min_evader = min(evader_dist)
        #     reward += -self.beta * (min_evader / self.observation_radius)
        # # If the episode is not over, get the closest evader in our observation radius. If no close evader found,
        # # then gives the largest penalty as a function of our observation radius.
        # if not all(evader_done_state):
        #     if evader_dist:
        #         min_evader = min(evader_dist)
        #     else:
        #         min_evader = self.observation_radius

        #     reward += -self.beta * ((min_evader / self.observation_radius))

        # reward += self.beta * (1 / (min_evader + 0.0000001))
        # reward += self.beta * ( 1 / ((min_evader - (agent.radius * 2)) ** 2 + 0.0000001))
        # reward += self.beta * max(
        #     min((1 / (min_evader - (agent.radius * 2) + 0.00000001)), 1), 0.0
        # )

        # penalty for collision with target
        if agent.collision_with(self.target):
            reward += -10
            agent.done = True
            self.target_done = True

        # # penalty for collision with obstacles
        if self.obstacles and self.obstacle_penalty:
            for obstacle in self.obstacles:
                if agent.collision_with(obstacle):
                    reward += -self.obstacle_penalty_weight

        # penalty for collision with other agents
        if len(self.agents) > 2 and self.agent_collision_penalty:
            for a in self.agents:
                if (a.id != agent.id) and (agent.collision_with(a)):
                    reward += -self.agent_penalty_weight

        # small reward for time
        # TODO: fix this if want to enable
        # reward += -self.time_step_penalty * self.time_step

        return reward

    def get_constraints(self):
        return {agent.id: self._get_agent_constraint(agent) for agent in self.agents}

    def get_constraint_margin(self):
        return {
            agent.id: self._get_agent_constraint_margin(agent) for agent in self.agents
        }

    def _get_agent_constraint_margin(self, agent):
        constraint_margin = []
        constraint_margin.append(agent.radius + self.target.radius)

        constraint_margin.extend(
            [
                agent.radius + other_agent.radius
                for other_agent in self.agents
                if other_agent.id != agent.id
            ]
        )

        constraint_margin.extend([agent.radius + ob.radius for ob in self.obstacles])

        constraint_margin = np.array(constraint_margin)

        norm_high_c = np.array([1] * self.num_constraints)
        norm_low_c = np.array([-1] * self.num_constraints)

        high_c = np.array([2 * self.target.radius] * self.num_constraints)
        low_c = np.array([0] * self.num_constraints)

        norm_c = (norm_high_c - norm_low_c) * (
            (constraint_margin - low_c) / (high_c - low_c)
        ) + norm_low_c

        return constraint_margin
        # return norm_c

    def _get_agent_constraint(self, agent):
        """Return simulation constraints"""
        constraints = []

        # # agent collision with target
        # constraints.append(agent.rel_dist(self.target))

        # agent collision with target
        constraints.append(
            (agent.radius + self.target.radius + self._constraint_slack)
            - agent.rel_dist(self.target)
        )

        # # # agent collision with target
        # constraints.extend(
        #     [
        #         agent.rel_dist(other_agent)
        #         for other_agent in self.agents
        #         if other_agent.id != agent.id
        #     ]
        # )

        # agent collision with other agents
        constraints.extend(
            [
                (agent.radius + other_agent.radius + self._constraint_slack)
                - agent.rel_dist(other_agent)
                for other_agent in self.agents
                if other_agent.id != agent.id
            ]
        )

        # agent collision with obstacles
        # constraints.extend([agent.rel_dist(ob) for ob in self.obstacles])

        # agent collision with obstacles
        constraints.extend(
            [
                (agent.radius + ob.radius + self._constraint_slack) - agent.rel_dist(ob)
                for ob in self.obstacles
            ]
        )

        return np.array(constraints)
        # return self._norm_constraints(np.array(constraints))

    def _norm_constraints(self, constraints):
        """Normalize constraints between -1 and 1
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        # x_{normalized} = (b-a)\frac{x - min(x)}{max(x) - min(x)} + a
        """

        norm_high_c = np.array([1] * self.num_constraints)
        norm_low_c = np.array([-1] * self.num_constraints)
        # high_c = np.array([2 * self.target.radius] * self.num_constraints)
        # low_c = np.array(
        #     [2 * self.target.radius - self.max_distance] * self.num_constraints
        # )

        high_c = np.array([self.max_distance] * self.num_constraints)
        low_c = np.array([0] * self.num_constraints)

        norm_c = (norm_high_c - norm_low_c) * (
            (constraints - low_c) / (high_c - low_c)
        ) + norm_low_c

        return norm_c

    def _get_done(self):
        """The simulation is done when the target is breach by any agents,
        pursuers or evaders and if all evaders are done.

        # https://github.com/ray-project/ray/issues/10761
        # reporting done multiple times for agent will cause error.
        Returns:
                [type]: [description]
        """
        done = {agent.id: agent.done for agent in self.agents}

        evader_done = [evader.done for evader in self.evaders]

        pursuer_done = [agent.done for agent in self.agents]

        # Done when:
        #   target is breached
        #   all pursuers are done, this is unlikely for now
        #   all the evaders are captured
        done["__all__"] = (
            self.target_done
            or all(pursuer_done)
            or all(evader_done)
            or self.time_elapse >= self.max_time
        )

        return done

    # normalize the obs space
    def _calc_obs(self, agent, norm=True):

        evader_states = []
        for evader in self.evaders:

            if not evader.done and agent.sensed(evader):
                evader_rel_state = []
                # evader_rel_state.append(int(evader.type))
                # evader_rel_state.append(int(agent.sensed(evader)))
                evader_rel_state.append(agent.rel_dist(evader))
                evader_rel_state.append(agent.rel_bearing_error(evader))
                evader_rel_state.append(agent.rel_bearing_entity_error(evader))
                evader_rel_state.append(evader.rel_dist(self.target))
                evader_rel_state.append(evader.rel_bearing_error(self.target))

                evader_states.append(evader_rel_state)
        evader_states = sorted(evader_states, key=lambda x: x[0])

        num_inactive_evaders = self.max_num_evaders - len(evader_states)

        last_evader_state = [
            # int(AgentType.E),
            # int(False),  # sensed
            0,  # relative distance
            0,  # relative bearing
            0,  # rel_bearing_self
            0,  # rel distance target
            0,  # rel bearing target
        ]

        # flatten the list of evaders
        evader_states = [state for sublist in evader_states for state in sublist]

        other_agent_states = []
        for other_agent in self.agents:

            # skip the agent were evaluating observation for
            if agent.id != other_agent.id and agent.sensed(other_agent):

                other_agent_rel_state = []

                # other_agent_rel_state.append(int(other_agent.type))
                # other_agent_rel_state.append(int(agent.sensed(other_agent)))

                # distance to other agent
                other_agent_rel_state.append(agent.rel_dist(other_agent))

                # bearing to other agent
                other_agent_rel_state.append(agent.rel_bearing_error(other_agent))

                # relative bearing of other agent to current agent
                other_agent_rel_state.append(
                    agent.rel_bearing_entity_error(other_agent)
                )

                # distance to target
                other_agent_rel_state.append(other_agent.rel_dist(self.target))

                # bearing to target
                other_agent_rel_state.append(other_agent.rel_bearing_error(self.target))

                other_agent_states.append(other_agent_rel_state)

        # sort the list by dist closest to agent by this method:
        # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
        other_agent_states = sorted(other_agent_states, key=lambda x: x[0])

        # don't include the main pursuer
        num_inactive_other_agents = self.max_num_pursuers - len(other_agent_states) - 1

        last_other_agent_state = [
            # int(AgentType.P),
            # int(False),  # can't sense this agent
            0,  # relative distance
            0,  # relative bearing
            0,  # relative bearing of the other agent
            0,  # relative distance to target
            0,  # relative bearing to target
        ]

        # flatten the list of other agent states
        other_agent_states = [
            state for sublist in other_agent_states for state in sublist
        ]

        obstacle_states = []

        for obstacle in self.obstacles:
            if agent.sensed(obstacle):
                obstacle_rel_state = []
                # obstacle_rel_state.append(int(obstacle.type))
                # obstacle_rel_state.append(int(agent.sensed(obstacle)))
                ob_dist = agent.rel_dist(obstacle)
                ob_dist = np.clip(ob_dist, 0, self.max_distance)
                obstacle_rel_state.append(ob_dist)
                obstacle_rel_state.append(agent.rel_bearing_error(obstacle))

                # add the obstacle to the obstacle list
                obstacle_states.append(obstacle_rel_state)

        obstacle_states = sorted(obstacle_states, key=lambda x: x[0])

        num_inactive_obstacles = self.max_num_obstacles - len(obstacle_states)

        last_obstacle_state = [
            # int(AgentType.O),  # obstacle type
            # int(False),  # obstacle sensed or not
            0,  # relative distance to agent
            0,  # relative bearing to agent
        ]

        # flatten list of obstacle states
        obstacle_states = [state for sublist in obstacle_states for state in sublist]

        obs = np.array(
            [
                *agent.state,
                agent.rel_dist(self.target),
                agent.rel_bearing_error(self.target),
                *evader_states,
                *(last_evader_state * num_inactive_evaders),
                *other_agent_states,
                *(last_other_agent_state * num_inactive_other_agents),
                *obstacle_states,
                *(last_obstacle_state * num_inactive_obstacles),
            ],
            dtype=np.float32,
        )

        raw_obs = np.copy(obs)
        if norm:
            obs = self.norm_obs_space(obs)

        agent_constraint = self._get_agent_constraint(agent)
        action_g, action_h, action_r = self.calc_projected_states(agent)
        action_a = self._scale_action(self.cum_a_bar[agent.id])
        # action_a = self._scale_action(np.array([agent.v, agent.w]))
        # action_a = self._scale_action(self.proj_safe_action(agent))

        obs_dict = {
            "observations": obs.astype(np.float32),
            "raw_obs": raw_obs.astype(np.float32),
            "constraints": agent_constraint.astype(np.float32),
            "action_g": action_g.astype(np.float32),
            "action_h": action_h.astype(np.float32),
            "action_r": action_r.astype(np.float32),
            # "action_a": action_a.astype(np.float32),
        }
        # return obs.astype(np.float32)
        return obs_dict

    def norm_obs_space(self, obs):

        """"""

        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        # x_{normalized} = (b-a)\frac{x - min(x)}{max(x) - min(x)} + a
        norm_orbs = (self.norm_high_obs - self.norm_low_obs) * (
            (obs - self.low_obs) / (self.high_obs - self.low_obs)
        ) + self.norm_low_obs

        return norm_orbs

    def _scale_action(self, action):
        """Scale agent action between default norm action values"""
        # assert np.all(np.greater_equal(action, self.low)), (action, self.low)
        # assert np.all(np.less_equal(action, self.high)), (action, self.high)
        action = (self.norm_high - self.norm_low) * (
            (action - self.low) / (self.high - self.low)
        ) + self.norm_low

        return action

    # TODO: use pygame instead of pyglet, https://www.geeksforgeeks.org/python-display-text-to-pygame-window/
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
    def render(self, mode="human"):
        # view colorcode: https://www.rapidtables.com/web/color/RGB_Color.html
        colors = {
            "black": (0, 0, 0),
            "red": (1, 0, 0),
            "orange": (1, 0.4, 0),
            "light_orange": (1, 178 / 255, 102 / 255),
            "yellow": (1, 1, 0),
            "light_yellow": (1, 1, 204 / 255),
            "green": (0, 1, 0),
            "blue": (0, 0, 1),
            "indigo": (0.2, 0, 1),
            "dark_gray": (0.2, 0.2, 0.2),
            "pursuer": (5 / 255, 28 / 255, 176 / 255),
            "evader": (240 / 255, 0, 0),
            "white": (1, 1, 1),
        }

        x_scale = self.screen_width / self.env_width
        y_scale = self.screen_height / self.env_height

        from cuas.envs import rendering

        if self.viewer is None:

            target = rendering.make_circle(self.target.radius * x_scale, filled=True)
            target.set_color(*colors["light_orange"])

            target_trans = rendering.Transform(
                translation=(
                    self.target.x * x_scale,
                    self.target.y * y_scale,
                ),
            )

            target.add_attr(target_trans)

            target_alarmed_rad = rendering.make_circle(
                self.target.alarmed_radius * x_scale, filled=True
            )
            target_alarmed_rad.set_color(*colors["light_yellow"])
            target_alarmed_rad.add_attr(target_trans)

            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            self.viewer.add_geom(target_alarmed_rad)
            self.viewer.add_geom(target)

            self.obstacle_transforms = []
            for obs in self.obstacles:
                obstacle = rendering.make_circle(obs.radius * x_scale, filled=True)
                obstacle_heading = rendering.Line((0, 0), (obs.radius * x_scale, 0))
                obstacle.set_color(*colors["black"])
                obstacle_heading.set_color(*colors["white"])

                obs_trans = rendering.Transform(
                    translation=(
                        obs.x * x_scale,
                        obs.y * y_scale,
                    )
                )
                obstacle.add_attr(obs_trans)
                obstacle_heading.add_attr(obs_trans)
                self.obstacle_transforms.append(obs_trans)

                self.viewer.add_geom(obstacle)
                self.viewer.add_geom(obstacle_heading)

            # create agents
            # this includes evader
            self.agent_transforms = []
            self.all_agents = []
            self.all_agents.extend(self.agents)
            self.all_agents.extend(self.evaders)

            for agent in self.all_agents:
                agent_rad = rendering.make_circle(agent.radius * x_scale, filled=False)
                agent_heading = rendering.Line((0, 0), ((agent.radius) * x_scale, 0))
                # add sensor
                agent_sensor = rendering.make_circle(
                    self.observation_radius * x_scale,
                )
                # opacity (0 = invisible, 1 = visible)
                agent_sensor.set_color(*colors["red"], 0.05)

                # TODO: enable this section if sensor is cone shape
                # fov_left = (
                #     self.observation_radius * np.cos(self.observation_fov / 2),
                #     self.observation_radius * np.sin(self.observation_fov / 2),
                # )

                # fov_right = (
                #     self.observation_radius * np.cos(-self.observation_fov / 2),
                #     self.observation_radius * np.sin(-self.observation_fov / 2),
                # )

                # agent_sensor = rendering.FilledPolygon(
                #     [
                #         (0, 0),
                #         (fov_left[0] * x_scale, fov_left[1] * y_scale),
                #         (fov_right[0] * x_scale, fov_right[1] * y_scale),
                #     ]
                # )
                # # opacity (0 = invisible, 1 = visible)
                # agent_sensor.set_color(*colors["red"], 0.25)

                if agent.type == AgentType.P:

                    agent_sprite = rendering.Image(
                        fname=str(RESOURCE_FOLDER / "pursuer.png"),
                        width=agent.radius * x_scale,
                        height=agent.radius * y_scale,
                    )
                    agent_rad.set_color(*colors["pursuer"])
                    agent_heading.set_color(*colors["pursuer"])

                else:
                    agent_sprite = rendering.Image(
                        fname=str(RESOURCE_FOLDER / "evader.png"),
                        width=agent.radius * x_scale,
                        height=agent.radius * y_scale,
                    )
                    agent_rad.set_color(*colors["evader"])
                    agent_heading.set_color(*colors["evader"])

                agent_transform = rendering.Transform(
                    translation=(agent.x * x_scale, agent.y * y_scale),
                    rotation=agent.theta,
                )

                self.agent_transforms.append(agent_transform)

                agent_sprite.add_attr(agent_transform)
                agent_rad.add_attr(agent_transform)
                agent_heading.add_attr(agent_transform)
                agent_sensor.add_attr(agent_transform)

                self.viewer.add_geom(agent_sprite)
                self.viewer.add_geom(agent_rad)
                self.viewer.add_geom(agent_heading)
                # TODO: evader should also have limited view of environment.
                if self.show_observation_radius and agent.type == AgentType.P:
                    self.viewer.add_geom(agent_sensor)

        for agent, agent_transform in zip(self.all_agents, self.agent_transforms):
            agent_transform.set_translation(agent.x * x_scale, agent.y * y_scale)
            agent_transform.set_rotation(agent.theta)

            if self.render_trace:
                agent_line = rendering.Line((0, 0), ((agent.radius) * x_scale, 0))
                if agent.type == AgentType.P:
                    agent_line.set_color(*colors["pursuer"])
                else:
                    agent_line.set_color(*colors["evader"])

                agent_transform = rendering.Transform(
                    translation=(agent.x * x_scale, agent.y * y_scale),
                    rotation=agent.theta,
                )
                agent_line.add_attr(agent_transform)
                self.viewer.add_geom(agent_line)

        for obstacle, obstacle_transform in zip(
            self.obstacles, self.obstacle_transforms
        ):
            obstacle_transform.set_translation(
                obstacle.x * x_scale, obstacle.y * y_scale
            )
            obstacle_transform.set_rotation(obstacle.theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
        self.viewer = None
