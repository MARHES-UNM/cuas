import os
import context
import cuas
import configparser
import unittest
import numpy as np
from cuas.envs import cuas_env_multi_agent

from cuas.envs.cuas_env_multi_agent import CuasEnvMultiAgent


class TestMultiAgent(unittest.TestCase):
    def setUp(self):
        self.cuas_multi_agent = CuasEnvMultiAgent()

    def test_config(self):
        config = {"env_width": 82, "env_height": 61}

        cuas_multi_agent = CuasEnvMultiAgent(config)

        self.assertEqual(config["env_width"], cuas_multi_agent.env_width)
        self.assertEqual(config["env_height"], cuas_multi_agent.env_height)

    def test_init(self):
        obs = self.cuas_multi_agent.reset()

        for observation in obs.values():
            self.assertTrue(np.all(observation >= -1))
            self.assertTrue(np.all(observation <= 1))
        self.assertFalse(all([value for _, value in self.cuas_multi_agent._get_done().items()])
            )

    def test_get_obs(self):

        for agent in self.cuas_multi_agent.agents:
            agent.x = 0
            agent.y = 0
            agent.theta = -np.pi

            agent.state = [0, 0]
            obs = self.cuas_multi_agent._calc_obs(agent)
            np.testing.assert_almost_equal([-1, -1, -1, -1, 0], obs[:5])

    def test_sorting_obs(self):
        agent_pos = [[0, 0, 0], [10, 10, 0], [20, 20, -np.pi], [30, 30, np.pi]]
        self.repos_agents(agent_pos)
        
        for id, agent in enumerate(self.cuas_multi_agent.agents):
            obs = self.cuas_multi_agent._calc_obs(agent)
            
            print(f"obs id: {id} order: {obs}")

    @unittest.skip
    def test_alpha_reward(self):
        agent_pos = [[0, 0, 0], [0, 55, 0], [75, 0, -np.pi], [75, 55, np.pi]]

        self.repos_agents(agent_pos)
        for agent in self.cuas_multi_agent.agents:
            rew = self.cuas_multi_agent._calc_reward(agent)
            self.assertEqual(0.0, rew)
            self.assertFalse(agent.done)

        agent_pos = [[0, 0, 0], [0, 55, 0], [75, 0, -np.pi], [55, 55, np.pi]]
        self.repos_agents(agent_pos)

        for agent in self.cuas_multi_agent.agents:
            rew = self.cuas_multi_agent._calc_reward(agent)
            self.assertGreaterEqual(rew, -1)
            self.assertLessEqual(rew, 0)
            self.assertFalse(agent.done)

    @unittest.skip
    def test_calc_reward(self):
        start_position = 0
        for agent in self.cuas_multi_agent.agents:
            start_position += 10
            agent.x = start_position
            agent.y = start_position

        for agent in self.cuas_multi_agent.agents:
            rew = self.cuas_multi_agent._calc_reward(agent)
            self.assertEqual(0.0, rew)
            self.assertFalse(agent.done)
            # print(f"agent done: {agent.done}")

        self.cuas_multi_agent.agents[0].x = 0
        self.cuas_multi_agent.agents[0].y = 0
        self.cuas_multi_agent.agents[2].x = 1
        self.cuas_multi_agent.agents[2].y = 1

        rew = {
            agent.id: self.cuas_multi_agent._calc_reward(agent)
            for agent in self.cuas_multi_agent.agents
        }
        # when evaders are done, agents shouldn't get any more reward
        self.assertEqual(0.25, rew[0])
        self.assertEqual(0, rew[1])
        self.assertEqual(-0.25, rew[2])
        self.assertEqual(0, rew[3])

        self.assertFalse(self.cuas_multi_agent.agents[0].done)
        self.assertTrue(self.cuas_multi_agent.agents[2].done)

        done = self.cuas_multi_agent._get_done()
        self.assertEqual(False, done[0])
        self.assertEqual(False, done[1])
        self.assertEqual(True, done[2])
        self.assertEqual(False, done[3])
        self.assertTrue(done["__all__"])

        rew = {
            agent.id: self.cuas_multi_agent._calc_reward(agent)
            for agent in self.cuas_multi_agent.agents
        }
        # when evaders are done, agents shouldn't get any more reward
        self.assertEqual(0, rew[0])
        self.assertEqual(0, rew[1])
        self.assertEqual(0, rew[2])
        self.assertEqual(0, rew[3])

        self.cuas_multi_agent.agents[3].x = 40
        self.cuas_multi_agent.agents[3].y = 30

        rew = {
            agent.id: self.cuas_multi_agent._calc_reward(agent)
            for agent in self.cuas_multi_agent.agents
        }

        # print(rew)
        self.assertEqual(-1, rew[0])
        self.assertEqual(-1, rew[1])
        self.assertEqual(0, rew[2])
        self.assertEqual(1, rew[3])
        # self.assertEqual(-1, self.cuas_multi_agent._calc_reward(self.cuas_multi_agent.agents[1], alpha=0))

        self.cuas_multi_agent.agents[1].x = 41
        self.cuas_multi_agent.agents[1].y = 31
        self.cuas_multi_agent.agents[3].done = False

        rew = {
            agent.id: self.cuas_multi_agent._calc_reward(agent)
            for agent in self.cuas_multi_agent.agents
        }

        # print(rew)
        self.assertEqual(-1, rew[0])
        self.assertEqual(-1.75, rew[1])
        self.assertEqual(0, rew[2])
        self.assertEqual(0.75, rew[3])

        done = self.cuas_multi_agent._get_done()
        self.assertEqual(False, done[0])
        self.assertEqual(True, done[1])
        self.assertEqual(True, done[2])
        self.assertEqual(True, done[3])
        self.assertTrue(done["__all__"])
        # self.assertEqual(-1.75, self.cuas_multi_agent._calc_reward(self.cuas_multi_agent.agents[1], alpha=0))
        # self.assertEqual(.75, self.cuas_multi_agent._calc_reward(self.cuas_multi_agent.agents[3], alpha=0))

        done = self.cuas_multi_agent._get_done()
        self.assertTrue(done["__all__"])

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
    def test_step(self):
        self.cuas_multi_agent = CuasEnvMultiAgent({"num_obstacles": 3})

        agent_pos = [
            [10, 30, 0],
            [20, 30, 0],
            [60, 30, -np.pi / 4],
            [70, 30, np.pi + np.pi / 4],
        ]

        self.repos_agents(agent_pos)

        actions = {i: np.array([0, 0]) for i in range(self.cuas_multi_agent.num_agents)}
        obs, rew, done, info = self.cuas_multi_agent.step(actions)
        self.cuas_multi_agent.render()
    
    # def test_other_agent_col(self):
    #     self.cuas_multi_agent = CuasEnvMultiAgent({"num_obstacles": 1})
        
    #     self.cuas_multi_agent.obstacle[0].x = 
    #     for obstacle in self.cuas_multi_agent.obstacles:
    #         obstacle.v = 0
    #         obstacle.w = 0
        
    #     agent_pos = [
    #         [10, 30, 0],
    #         [20, 30, 0],
    #         [60, 30, -np.pi / 4],
    #         [70, 30, np.pi + np.pi / 4],
    #     ]
    #     self.repos_agents(agent_pos)
        
        # print(obs, rew, done, info)

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

    # @unittest.skip
    def test_render(self):
        self.cuas_multi_agent = CuasEnvMultiAgent({"num_obstacles": 5, "evader_move_type": "repulsive"})
        for i in range(1000):
            actions = {
                i: np.array([-1, 0]) for i in range(self.cuas_multi_agent.num_agents)
            }
            self.cuas_multi_agent.step(actions)
            self.cuas_multi_agent.render()

    def test_unscale_action(self):
        action = np.array([-2, -3])
        with self.assertRaises(Exception) as context:
            self.cuas_multi_agent._unscale_action(action)
        action = np.array([0, -3])
        with self.assertRaises(Exception) as context:
            self.cuas_multi_agent._unscale_action(action)

        action = np.array([0, 3])
        with self.assertRaises(Exception) as context:
            self.cuas_multi_agent._unscale_action(action)

        action = np.array([-2, 1])
        with self.assertRaises(Exception) as context:
            self.cuas_multi_agent._unscale_action(action)

        action = np.array([0.5, 0.1])
        new_action = self.cuas_multi_agent._unscale_action(action)
        np.testing.assert_almost_equal(
            np.array(
                [
                    (action[0] + 1) * self.cuas_multi_agent.agent_v_max / 2,
                    action[1] * self.cuas_multi_agent.agent_w_max,
                ]
            ),
            new_action,
        )


if __name__ == "__main__":
    unittest.main()