import os
import pathlib
import gym
import numpy as np
from cuas import util
from cuas.agents.cuas_agents import Agent, AgentType, CuasAgent, Entity, Obstacle
from cuas.envs.base_cuas_env import BaseCuasEnv
from gym import spaces
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from cuas.policy.velocity_obstacle import compute_velocity

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
# RESOURCE_FOLDER = pathlib.Path("resources")
RESOURCE_FOLDER = path.joinpath("../../resources")


class CuasEnvMultiAgent(BaseCuasEnv, MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        self.viewer = None

        if env_config:
            self.config = env_config
        else:
            self.config = {}
        self._parse_config()
        self.time_elapse = 0
        self.seed(seed=self.sim_seed)
        self.target_done = False
        self.max_num_agents = 20
        self.max_num_obstacles = 20
        self.inactive_agents = self.max_num_agents - self.num_agents

        assert (
            self.inactive_agents >= 0
        ), f"{self.max_num_agents}:{self.num_agents}, max number of agents can't be less than number of agents"

        self.inactive_obstacles = self.max_num_obstacles - self.num_obstacles
        assert (
            self.inactive_obstacles >= 0
        ), f"{self.max_obstacles}:{self.num_obstales}, max number of obstacles can't be less than number of obstacles"
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()
        self.reset()

    def _parse_config(self):
        self.env_width = self.config.get("env_width", 80)
        self.env_height = self.config.get("env_height", 60)
        self.screen_width = self.config.get("screen_width", 800)
        self.screen_height = self.config.get("screen_height", 600)
        self.num_evaders = self.config.get("num_evaders", 1)
        self.num_pursuers = self.config.get("num_pursuers", 3)
        self.num_obstacles = self.config.get("num_obstacles", 0)
        self.sim_seed = self.config.get("seed", None)
        self.norm_env = self.config.get("normalize", True)

        self.agent_radius = self.config.get("agent_radius", 2)
        self.agent_v_min = self.config.get("v_min", 0)
        self.agent_v_max = self.config.get("v_max", 10)  # m/s
        self.agent_w_min = self.config.get("w_min", -np.pi * 2)  # rad/s
        self.agent_w_max = self.config.get("w_max", np.pi * 2)

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

        self.norm_low = np.array([-1, -1])
        self.norm_high = np.array([1, 1])

        self.low = np.array([self.agent_v_min, self.agent_w_min])
        self.high = np.array([self.agent_v_max, self.agent_w_max])

        self.target = Entity(
            x=self.config.get("target_x", 40),
            y=self.config.get("target_y", 30),
            r=self.config.get("target_radius", 5),
        )

        self.target.alarmed_radius = self.config.get("alarmed_radius", 30)
        self.enable_vo = self.config.get("enable_vo", False)

        self.num_agents = self.num_pursuers + self.num_evaders

    def _add_obstacles(self):
        def get_random_obstacle():
            x = np.random.random() * self.env_width
            y = np.random.random() * self.env_height
            theta = np.random.random() * np.pi
            r = self.obstacle_radius

            return Obstacle(x, y, theta, r)

        for _ in range(self.num_obstacles):
            temp_obstacle = get_random_obstacle()
            temp_obstacle.v = self.obstacle_v
            self.obstacles.append(temp_obstacle)

    def _create_agents(self, start_idx, agent_type, num_agents):
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

            return False

        def get_random_agent(id):
            x = np.random.random() * self.env_width
            y = np.random.random() * self.env_height
            theta = np.random.random() * np.pi
            r = self.agent_radius
            move_type = (
                self.pursuer_move_type
                if agent_type == AgentType.P
                else self.evader_move_type
            )

            return CuasAgent(id, agent_type, x, y, theta, r=r, move_type=move_type)

        agent_id = start_idx
        for _ in range(num_agents):
            in_collision = True
            # make sure agents in start in correct positions
            while in_collision:
                agent = get_random_agent(agent_id)
                in_collision = is_collision(agent)

            self.agents.append(agent)
            agent_id += 1

    def _get_observation_space(self):
        """"""
        # agent_state: x, y, theta, v, w, dist_target, theta_target
        self.norm_low_obs = [-1] * 7
        # evader_state: dist_evader, theta_evader, evader_rel_bearing, evader_dist_target, evader_rel_bearing_target
        self.norm_low_obs.extend([-1] * 5)
        # other_agents: dist_agent_j, theta_agent_j, agent_j_rel_bearing, agent_j_dist_target, agent_j_rel_bearing_target
        self.norm_low_obs.extend(([-1] * 5) * (self.max_num_agents - 2))
        # obstacle: distance_obstacle_k, theta_obstacle_k
        self.norm_low_obs.extend(([-1] * 2) * (self.max_num_obstacles))
        self.norm_low_obs = np.array(self.norm_low_obs)
        self.norm_high_obs = np.array([1] * self.norm_low_obs.size)

        self.max_distance = util.distance(point_2=(self.env_width, self.env_height))

        self.low_obs = [
            0,
            0,
            -np.pi,
            self.agent_v_min,
            self.agent_w_min,
            0,
            -np.pi,
        ]
        self.low_obs.extend([0, -np.pi, -np.pi, 0, -np.pi] * (self.max_num_agents - 1))

        self.low_obs.extend([0, -np.pi] * (self.max_num_obstacles))

        self.low_obs = np.array(self.low_obs)

        self.high_obs = [
            self.env_width,
            self.env_height,
            np.pi,
            self.agent_v_max,
            self.agent_w_max,
            self.max_distance,
            np.pi,
        ]

        self.high_obs.extend(
            [
                self.max_distance,
                np.pi,
                np.pi,
                self.max_distance,
                np.pi,
            ]
            * (self.max_num_agents - 1)
        )
        self.high_obs.extend([self.max_distance, np.pi] * (self.max_num_obstacles))

        self.high_obs = np.array(self.high_obs)

        return spaces.Dict(
            {
                i: spaces.Box(
                    low=self.norm_low_obs, high=self.norm_high_obs, dtype=np.float32
                )
                for i in range(self.num_agents)
            }
        )

    def _get_action_space(self):
        """"""
        return spaces.Dict(
            {
                i: spaces.Box(low=self.norm_low, high=self.norm_high, dtype=np.float32)
                for i in range(self.num_agents)
            }
        )

    def reset(self):
        """
        Reset environment
        """
        if self.viewer is not None:
            self.close()

        self.agents = []
        self.dones = set()
        self.obstacles = []
        self._add_obstacles()
        self._create_agents(0, AgentType.P, self.num_pursuers)
        self._create_agents(0 + self.num_pursuers, AgentType.E, self.num_evaders)
        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}
        self.time_elapse = 0
        self.target_done = False

        return obs

    def seed(self, seed=None):
        """ Random value to seed"""
        np.random.seed(seed)

        seed = seeding.np_random(seed)
        return [seed]

    def repulsive_action(self, agent):
        """https://link.springer.com/article/10.1007/s10514-020-09945-6"""
        kr = 10
        # ka should be 10
        ka = 10 * kr

        target = self.agents[-1] if agent.type == AgentType.P else self.target

        dist_to_target = agent.rel_dist(target)

        if dist_to_target >= 0.05:

            des_vx = (
                ka
                * (1 / (dist_to_target ** self.evader_alpha))
                * ((target.x - agent.x) / dist_to_target)
            )
            des_vy = (
                ka
                * (1 / (dist_to_target ** self.evader_alpha))
                * ((target.y - agent.y) / dist_to_target)
            )

            rep_vx = 0
            rep_vy = 0
            for other_agent in self.agents:
                if other_agent.id == agent.id:
                    continue

                dist_to_other_agent = agent.rel_dist(other_agent)
                rep_vx += (1 / (dist_to_other_agent ** self.evader_alpha)) * (
                    (other_agent.x - agent.x) / dist_to_other_agent
                )

                rep_vy += (1 / (dist_to_other_agent ** self.evader_alpha)) * (
                    (other_agent.y - agent.y) / dist_to_other_agent
                )

            for obs in self.obstacles:
                dist_to_obstacle = agent.rel_dist(obs)
                rep_vx += (1 / (dist_to_obstacle ** self.evader_alpha)) * (
                    (obs.x - agent.x) / dist_to_obstacle
                )

                rep_vy += (1 / (dist_to_obstacle ** self.evader_alpha)) * (
                    (obs.y - agent.y) / dist_to_obstacle
                )

            if agent.type == AgentType.P:
                dist_to_target = agent.rel_dist(self.target)
                rep_vx += (1 / (dist_to_target ** self.evader_alpha)) * (
                    (self.target.x - agent.x) / dist_to_target
                )

                rep_vy += (1 / (dist_to_target ** self.evader_alpha)) * (
                    (self.target.y - agent.y) / dist_to_target
                )

            des_vx += -kr * rep_vx
            des_vy += -kr * rep_vy
            des_v = self.agent_v_max * np.array([des_vx, des_vy]) / np.linalg.norm([des_vx, des_vy])

            dxu = self.si_uni_dyn(agent, des_v)
            # dx_dy = des_v * .01

            # des_theta = np.arctan2(dx_dy[1], dx_dy[0])
            # omega = 10 * (agent.theta - des_theta)
            # v_norm = np.linalg.norm(des_v)
            # # dxu =np.array( [v_norm, omega])
        else:
            dxu = np.array([0, 0])

        return dxu

    # TODO: fix this
    def go_to_goal(self, agent):
        """Policy for Evader to move to goal"""
        # https://asl.ethz.ch/education/lectures/autonomous_mobile_robots/spring-2020.html
        k_rho = 0.5
        k_alpha = 2
        k_beta = -0.01
        target = self.target if agent.type == AgentType.E else self.agents[-1]
        rho = agent.rel_dist(target)
        alpha = -agent.theta + agent.rel_bearing(target)
        beta = -agent.theta - alpha
        vw = np.array([k_rho * rho, k_alpha * alpha + k_beta * beta])
        vw = np.clip(
            vw,
            [self.agent_v_min, self.agent_w_min],
            [self.agent_v_max, self.agent_w_max],
        )

        agent.state = vw

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

    def _unscale_action(self, action):
        """[summary]

        Args:
            action ([type]): [description]

        Returns:
            [type]: [description]
        """
        # print("action: ", action)
        # print("action type:", type(action))
        # unnormalize the action
        assert np.all(np.greater_equal(action, self.norm_low)), (action, self.norm_low)
        assert np.all(np.less_equal(action, self.norm_high)), (action, self.norm_high)
        action = self.low + (self.high - self.low) * (
            (action - self.norm_low) / (self.norm_high - self.norm_low)
        )
        action = np.clip(action, self.low, self.high)

        return action

    # todo: caught agents shouldn't move
    def step(self, actions):
        """[summary]

        action is of type dictionary
        Args:
            action ([type], optional): [description]. Defaults to None.
        """

        self.time_elapse += self.time_step
        obs, rew, done, info = {}, {}, {}, {}

        for i, action in actions.items():

            if self.agents[i].done or self.agents[i].move_type == "static":
                self.agents[i].state = np.array([0, 0])

            elif self.agents[i].move_type == "rl":
                action = self._unscale_action(action)
                if self.enable_vo and self.agents[i].type == AgentType.P:
                    # TODO: troubleshoot why we get exception
                    try:
                        des_v = compute_velocity(
                            self.agents[i],
                            self.agents,
                            self.obstacles,
                            action,
                            agent_v_max=self.agent_v_max,
                            obstacle_v_max=self.obstacle_v,
                        )

                        # print("original action: ", action)
                        # action = des_v
                        action = self.si_uni_dyn(
                            self.agents[i], des_v, projection_distance=0.05
                        )

                        # print("updated action: ", action)
                    except Exception as e:
                        print("couldn't set VO, defaulting to original action")

                self.agents[i].state = action

            elif self.agents[i].move_type == "repulsive":
                action = self.repulsive_action(self.agents[i])

                if self.enable_vo and self.agents[i].type == AgentType.P:
                    # TODO: troubleshoot why we get exception
                    try:
                        des_v = compute_velocity(
                            self.agents[i],
                            self.agents,
                            self.obstacles,
                            action,
                            agent_v_max=self.agent_v_max,
                            obstacle_v_max=self.obstacle_v,
                        )

                        # print("original action: ", action)
                        # action = des_v
                        action = self.si_uni_dyn(
                            self.agents[i], des_v, projection_distance=0.05
                        )

                        # print("updated action: ", action)
                    except Exception as e:
                        print("couldn't set VO, defaulting to original action")
                self.agents[i].state = action

            elif self.agents[i].move_type == "random":
                self.agents[i].state = self._unscale_action(
                    self.action_space[0].sample()
                )
            elif self.agents[i].move_type == "go_to_goal":
                self.go_to_goal(self.agents[i])

        obs = {agent.id: self._calc_obs(agent) for agent in self.agents}
        rew = {agent.id: self._calc_reward(agent) for agent in self.agents}
        done = self._get_done()
        info = self._calc_info()

        for obstacle in self.obstacles:
            obstacle.step()

        return obs, rew, done, info

    def _calc_info(self):
        info = {}
        for agent in self.agents:
            target_collision = 0
            obstacle_collision = 0
            agent_collision = 0

            target_collision = 1 if agent.collision_with(self.target) else 0

            for other_agent in self.agents:
                if agent.id == other_agent.id:
                    continue

                # don't report pursuer to evader collision
                if agent.type == AgentType.P and other_agent.type == AgentType.E:
                    continue
                else:
                    agent_collision += 1 if agent.collision_with(other_agent) else 0

            for obstacle in self.obstacles:
                obstacle_collision += 1 if agent.collision_with(obstacle) else 0

            info[agent.id] = {
                "target_collision": target_collision,
                "agent_collision": agent_collision,
                "obstacle_collision": obstacle_collision,
            }

            if agent.type == AgentType.E:
                info[agent.id]["evader_captured"] = 1 if agent.captured else 0
        return info
        # return {agent.id: {} for agent in self.agents}

    def _calc_reward(self, agent):
        reward = 0

        if agent.type == AgentType.P:

            # Evader Reward/Penalty
            # Reward for reaching the evader
            evader = self.agents[-1]

            # else:
            dist_evader = agent.rel_dist(evader)

            # # TODO: change this to evader_dist if training doesn't work out
            # reward += -self.beta * (min_d / self.max_distance)
            # reward += -self.beta * (dist_evader / self.max_distance)

            e_capt = [
                a.collision_with(evader) for a in self.agents if a.type != AgentType.E
            ]

            mean_d = np.mean(
                [a.rel_dist(evader) for a in self.agents if a.type != AgentType.E]
            )
            min_d = min(
                [a.rel_dist(evader) for a in self.agents if a.type != AgentType.E]
            )
            if agent.collision_with(evader):
                reward += 1
            # if any(e_capt):
            # reward += 1
            else:
                reward += -self.beta * (
                    # (min_d / self.max_distance) + (dist_evader / self.max_distance)
                    (dist_evader / self.max_distance)
                )

            # penalty for evader reaching target
            evader_dist_target = evader.rel_dist(self.target)
            if evader.collision_with(self.target):
                reward += -1
            elif (
                not evader.collision_with(self.target)
                and evader_dist_target <= self.target.alarmed_radius
            ):
                reward += -self.alpha * (
                    1 - (evader_dist_target / self.target.alarmed_radius)
                )

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
                # min_obstacle_dist = min([agent.rel_dist(o) for o in self.obstacles])
                # collision_radius = agent.radius + self.obstacle_radius

                # if min_obstacle_dist < collision_radius:
                #     reward += -self.penalty_weight * (
                #         1 - (min_obstacle_dist / collision_radius)
                #     )

            # penalty for collision with other agents
            if len(self.agents) > 2 and self.agent_collision_penalty:
                for a in self.agents:
                    if a.type == AgentType.P and a.id != agent.id:
                        if agent.collision_with(a):
                            reward += -self.agent_penalty_weight
                # min_other_agent_dist = min(
                #     [
                #         agent.rel_dist(a)
                #         for a in self.agents
                #         if a.type == AgentType.P and a.id != agent.id
                #     ]
                # )
                # collision_radius = agent.radius * 2
                # if min_other_agent_dist < collision_radius:
                #     reward += -self.penalty_weight * (
                #         1 - (min_other_agent_dist / collision_radius)
                #     )

        # agent must be evader then
        else:
            # don't get any more reward for seeking evader that's done
            if agent.done:
                return reward

            if agent.collision_with(self.target):
                reward += 1
                agent.done = True
                self.target_done = True

            if self.obstacles and self.obstacle_penalty:
                for obstacle in self.obstacles:
                    if agent.collision_with(obstacle):
                        reward += -self.obstacle_penalty_weight

            for other_agent in self.agents:
                if other_agent.id != agent.id and agent.collision_with(other_agent):
                    reward += -1

                    if other_agent.type == AgentType.P:
                        agent.done = True
                        agent.captured = True
        return reward

    # todo: complete done
    def _get_done(self):
        """The simulation is done when the target is breach by any agents,pursuers or evaders and if all evaders are done.

        Returns:
            [type]: [description]
        """
        done = {}

        evader_done = []
        # target_collision = []
        pursuer_done = []
        all_done = []
        for idx, agent in enumerate(self.agents):
            done[idx] = agent.done

            # https://github.com/ray-project/ray/issues/10761
            # reporting done multiple times for agent will cause error.
            all_done.append(agent.done)
            if not agent.reported_done:
                done[idx] = agent.done
                agent.reported_done = True

            if agent.type == AgentType.E:
                evader_done.append(agent.done)

            else:
                pursuer_done.append(agent.done)

        # TODO: fix so that this ends if only target is breached or all evaders done. right now this ends because any(all_done) when one evader is
        # pursuer is done only if it it's the target
        # TODO: delete all done for now, actually delete agents when they are done
        done["__all__"] = (
            any(all_done)
            or self.target_done
            or any(pursuer_done)
            or all(evader_done)
            or self.time_elapse >= self.max_time
        )

        return done

    # normalize the obs space
    def _calc_obs(self, agent, norm=True):
        d_i_target = agent.rel_dist(self.target)
        theta_i_target = agent.rel_bearing_error(self.target)

        evader_state = []
        if agent.type != AgentType.E:
            evader = self.agents[-1]
            d_i_evader = agent.rel_dist(evader)
            theta_i_evader = agent.rel_bearing_error(evader)
            evader_rel_bearing = agent.rel_bearing_entity_error(evader)
            evader_d_target = evader.rel_dist(self.target)
            evader_theta_target = evader.rel_bearing_error(self.target)

            evader_state.extend(
                [
                    d_i_evader,
                    theta_i_evader,
                    evader_rel_bearing,
                    evader_d_target,
                    evader_theta_target,
                ]
            )

        other_agent_state = []
        for other_agent in self.agents:
            if agent.id == other_agent.id or other_agent.type == AgentType.E:
                continue

            other_agent_rel_state = []
            # distance to other agent
            other_agent_rel_state.append(agent.rel_dist(other_agent))

            # bearing to other agent
            other_agent_rel_state.append(agent.rel_bearing_error(other_agent))

            # relative bearing of other agent to current agent
            other_agent_rel_state.append(agent.rel_bearing_entity_error(other_agent))

            # distance to target
            other_agent_rel_state.append(other_agent.rel_dist(self.target))

            # bearing to target
            other_agent_rel_state.append(other_agent.rel_bearing_error(self.target))

            other_agent_state.append(other_agent_rel_state)

        # sort the list by dist closest to agent by this method:
        # https://www.geeksforgeeks.org/python-program-to-sort-a-list-of-tuples-by-second-item/
        other_agent_state = sorted(other_agent_state, key=lambda x: x[0])

        # case of one pursuer, one evader
        if len(self.agents) <= 2:
            last_agent_state = [
                self.max_distance,
                np.pi,
                np.pi,
                self.max_distance,
                np.pi,
            ]
        # case of more than one pursuer
        else:
            last_agent_state = other_agent_state[-1]
        # flatten the list
        other_agent_state = [
            state for sublist in other_agent_state for state in sublist
        ]

        obstacle_states = []
        # when number of obstacles is > 0
        if self.obstacles:
            for obstacle in self.obstacles:
                obstacle_rel_state = []
                ob_dist = agent.rel_dist(obstacle)
                ob_dist = np.clip(ob_dist, 0, self.max_distance)
                obstacle_rel_state.append(ob_dist)
                obstacle_rel_state.append(agent.rel_bearing_error(obstacle))
                obstacle_states.append(obstacle_rel_state)

            obstacle_states = sorted(obstacle_states, key=lambda x: x[0])
            last_obstacle_state = obstacle_states[-1]

            obstacle_states = [
                state for sublist in obstacle_states for state in sublist
            ]

        # when there's no obstacle
        else:
            last_obstacle_state = [self.max_distance, np.pi]
        obs = np.array(
            [
                *agent.state,
                d_i_target,
                theta_i_target,
                *evader_state,
                *other_agent_state,
                *(last_agent_state * self.inactive_agents),
                *obstacle_states,
                *(last_obstacle_state * self.inactive_obstacles),
            ]
        )

        if norm:
            obs = self.norm_obs_space(obs)

        return obs

    def norm_obs_space(self, obs):

        """"""

        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        # x_{normalized} = (b-a)\frac{x - min(x)}{max(x) - min(x)} + a
        norm_orbs = (self.norm_high_obs - self.norm_low_obs) * (
            (obs - self.low_obs) / (self.high_obs - self.low_obs)
        ) + self.norm_low_obs

        return norm_orbs

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
            "white": (1,1,1),
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
            self.agent_transforms = []
            for agent in self.agents:
                agent_rad = rendering.make_circle(agent.radius * x_scale, filled=False)
                agent_heading = rendering.Line((0, 0), ((agent.radius) * x_scale, 0))

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

                self.viewer.add_geom(agent_sprite)
                self.viewer.add_geom(agent_rad)
                self.viewer.add_geom(agent_heading)

        for agent, agent_transform in zip(self.agents, self.agent_transforms):
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
