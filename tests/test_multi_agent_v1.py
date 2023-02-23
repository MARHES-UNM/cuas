import os
import context
import cuas
import configparser
import unittest
import numpy as np
from cuas.envs import cuas_env_multi_agent_v1

from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1


class TestMultiAgent(unittest.TestCase):
    def setUp(self):
        self.cuas_multi_agent = CuasEnvMultiAgentV1()

    def test_init(self):
        obs = self.cuas_multi_agent.reset()

        for observation in obs.values():
            self.assertTrue(np.all(observation >= -1))
            self.assertTrue(np.all(observation <= 1))

        # no agent should be done
        self.assertFalse(
            all([value for _, value in self.cuas_multi_agent._get_done().items()])
        )

    # def test_norm_constraints(self):
    #     for _ in range(20):
    #         actions = self.cuas_multi_agent.action_space.sample()

    #         obs, rew, done, info = self.cuas_multi_agent.step(actions)
    #         constraints = self.cuas_multi_agent.get_constraints()

    #         for c in constraints.values():
    #             self.assertTrue(
    #                 np.all(np.greater_equal(c, -1)),
    #                 f"\nvalue: {c} not in range: \n\t-1",
    #             )

    #             self.assertTrue(
    #                 np.all(np.less_equal(c, 1)), f"\nvalue: {c} not in range:\n\t1"
    #             )

    def test_config(self):
        config = {"env_width": 82, "env_height": 61}

        cuas_multi_agent = CuasEnvMultiAgentV1(config)

        self.assertEqual(config["env_width"], cuas_multi_agent.env_width)
        self.assertEqual(config["env_height"], cuas_multi_agent.env_height)

    def test_observation_space(self):
        obs_space = self.cuas_multi_agent.observation_space.sample()
        self.assertEqual(len(obs_space), self.cuas_multi_agent.num_agents)

        agent_1_space = obs_space[0]
        self.assertEqual(
            len(agent_1_space),
            self.cuas_multi_agent.agent_num_states
            + (
                self.cuas_multi_agent.evader_other_agent_num_states
                * self.cuas_multi_agent.max_num_evaders
            )
            + (
                self.cuas_multi_agent.evader_other_agent_num_states
                * (self.cuas_multi_agent.max_num_agents - 1)
            )
            + (
                self.cuas_multi_agent.obs_num_states
                * self.cuas_multi_agent.max_num_obstacles
            ),
        )

        for _ in range(20):
            obs = self.cuas_multi_agent.observation_space.sample()

            for observation in obs.values():
                self.assertTrue(np.all(observation >= -1))
                self.assertTrue(np.all(observation <= 1))

    def test_local_obs(self):
        config = {
            "env_width": 82,
            "env_height": 61,
            "observation_type": "local",
            "observation_radius": 50,
        }

        cuas_multi_agent = CuasEnvMultiAgentV1(config)

        obs = cuas_multi_agent.reset()

        cuas_multi_agent.render()

        print(obs)

    def test_space(self):
        actions = self.cuas_multi_agent.action_space.sample()

        for i, action in actions.items():
            self.assertTrue(
                np.all(np.greater_equal(action, self.cuas_multi_agent.norm_low))
            )
            self.assertTrue(
                np.all(np.less_equal(action, self.cuas_multi_agent.norm_high))
            )

        expected_obs_type = self.cuas_multi_agent.observation_space[0]
        obs = self.cuas_multi_agent.reset()

        for observation in obs.values():
            self.assertTrue(
                np.all(np.greater_equal(observation, expected_obs_type.low)),
                f"\nvalue: {observation} not in range:\n\t{expected_obs_type.low}",
            )

            self.assertTrue(
                np.all(np.less_equal(observation, expected_obs_type.high)),
                f"\nvalue: {observation} not in range:\n\t{expected_obs_type.high}",
            )

        for _ in range(20):
            actions = self.cuas_multi_agent.action_space.sample()

            obs, rew, done, info = self.cuas_multi_agent.step(actions)

            for observation in obs.values():
                self.assertTrue(
                    np.all(np.greater_equal(observation, expected_obs_type.low)),
                    f"\nvalue: {observation} not in range:\n\t{expected_obs_type.low}",
                )

                self.assertTrue(
                    np.all(np.less_equal(observation, expected_obs_type.high)),
                    f"\nvalue: {observation} not in range:\n\t{expected_obs_type.high}",
                )

    def test_constraint(self):
        num_obstacles = 2
        num_pursuers = 4
        self.cuas_multi_agent = CuasEnvMultiAgentV1(
            {
                "num_obstacles": num_obstacles,
                "num_evaders": 1,
                "num_pursuers": num_pursuers,
            }
        )
        c_margin = {
            i: np.zeros(num_pursuers + num_obstacles)
            for i in range(self.cuas_multi_agent.num_pursuers)
        }
        agent_pos = [[0, 0, 0], [0, 55, 0], [75, 0, -np.pi], [75, 55, np.pi]]

        self.repos_agents(agent_pos)
        self.cuas_multi_agent.render()

        c = self.cuas_multi_agent.get_constraints()
        # c_margin = self.cuas_multi_agent.get_constraint_margin()

        for (_, c_agent), (_, c_margin_agent) in zip(c.items(), c_margin.items()):
            self.assertTrue(
                np.all(np.less(c_agent, c_margin_agent)),
                f"\nvalue: {c} not less than:\n\t{c_margin}",
            )

        agent_pos = [[0, 0, 0], [0, 3, 0], [75, 0, -np.pi], [75, 1, np.pi]]

        self.repos_agents(agent_pos)
        self.cuas_multi_agent.render()

        c = self.cuas_multi_agent.get_constraints()
        # c_margin = self.cuas_multi_agent.get_constraint_margin()
        c = np.concatenate([v for k, v in sorted(c.items())], 0)
        c_margin = np.concatenate([v for k, v in sorted(c_margin.items())], 0)

        self.assertFalse(
            np.all(np.less(c, c_margin)),
            f"\nvalue: {c} not less than: \n\t{c_margin}",
        )

        agent_pos = [[0, 0, 0], [0, 3, 0], [75, 0, -np.pi], [75, 4, np.pi]]

        self.repos_agents(agent_pos)
        self.cuas_multi_agent.render()

        c = self.cuas_multi_agent.get_constraints()
        c = np.concatenate([v for k, v in sorted(c.items())], 0)

        self.assertTrue(
            np.all(np.less(c, c_margin)),
            f"\nvalue: {c} not less than: \n\t{c_margin}",
        )

        # constraint with target
        agent_pos = [[0, 0, 0], [34, 30, 0], [75, 0, -np.pi], [75, 4, np.pi]]

        self.repos_agents(agent_pos)
        self.cuas_multi_agent.render()

        c = self.cuas_multi_agent.get_constraints()
        c = np.concatenate([v for k, v in sorted(c.items())], 0)

        self.assertFalse(
            np.all(np.less(c, c_margin)),
            f"\nvalue: {c} not less than: \n\t{c_margin}",
        )

        for _ in range(10):
            self.cuas_multi_agent.render()

    def repos_agents(self, agent_pos):

        self.cuas_multi_agent.agents[0].x = agent_pos[0][0]
        self.cuas_multi_agent.agents[0].y = agent_pos[0][1]
        self.cuas_multi_agent.agents[0].theta = agent_pos[0][2]

        self.cuas_multi_agent.agents[1].x = agent_pos[1][0]
        self.cuas_multi_agent.agents[1].y = agent_pos[1][1]
        self.cuas_multi_agent.agents[1].theta = agent_pos[1][2]

        self.cuas_multi_agent.agents[2].x = agent_pos[2][0]
        self.cuas_multi_agent.agents[2].y = agent_pos[2][1]
        self.cuas_multi_agent.agents[2].theta = agent_pos[2][2]

        self.cuas_multi_agent.agents[3].x = agent_pos[3][0]
        self.cuas_multi_agent.agents[3].y = agent_pos[3][1]
        self.cuas_multi_agent.agents[3].theta = agent_pos[3][2]

    @unittest.skip
    def test_render(self):
        self.cuas_multi_agent = CuasEnvMultiAgentV1(
            {
                "num_obstacles": 5,
                "evader_move_type": "repulsive",
                "num_evaders": 1,
                "num_pursuers": 4,
                "obstacle_radius": 4,
                "obstacle_v": 5,
                "render_trace": False,
                "seed": 123,
                "evader_alpha": 1,
            }
        )
        for i in range(1000):
            actions = self.cuas_multi_agent.action_space.sample()

            self.cuas_multi_agent.step(actions)
            self.cuas_multi_agent.render()
            if self.cuas_multi_agent._get_done()["__all__"]:
                self.cuas_multi_agent.reset()

    @unittest.skip
    def test_safe_action(self):
        self.cuas_multi_agent = CuasEnvMultiAgentV1(
            {
                "num_obstacles": 5,
                "evader_move_type": "repulsive",
                "num_evaders": 1,
                "num_pursuers": 4,
                "obstacle_radius": 4,
                "obstacle_v": 5,
                "render_trace": False,
                "seed": 123,
                "evader_alpha": 1,
                "use_safe_action": True
            }
        )
        for i in range(5000):
            actions = self.cuas_multi_agent.action_space.sample()

            self.cuas_multi_agent.step(actions)
            self.cuas_multi_agent.render()
            if self.cuas_multi_agent._get_done()["__all__"]:
                self.cuas_multi_agent.reset()


if __name__ == "__main__":
    unittest.main()