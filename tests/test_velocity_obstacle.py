import unittest
import context
import numpy as np
from cuas.agents.cuas_agents import AgentType

from cuas.policy.velocity_obstacle import compute_velocity
from cuas.envs.cuas_env_multi_agent import CuasEnvMultiAgent


class TestVelocityObstacle(unittest.TestCase):
    def setUp(self):
        self.config = {
            "num_pursuers": 4,
            "num_evaders": 1,
            "num_obstacles": 1,
            "seed": 40,
            "obstacle_penalty": True,
            "agent_collision_penalty": True,
            "agent_radius": 1,
            "obstacle_radius": 1,
            "obstacle_v": 1,
            # "evader_move_type": "static",
            "pursuer_move_type": "repulsive",
            "enable_vo": True,
        }

        self.cuas_multi_agent = CuasEnvMultiAgent(self.config)
    
    def test_vo_1p1o(self):
        self.config.update({"num_pursuers": 1})
        self.cuas_multi_agent = CuasEnvMultiAgent(self.config)
        
        self.cuas_multi_agent.reset()
        
        self.cuas_multi_agent.agents[0].x = 10
        self.cuas_multi_agent.agents[0].y = 10
        self.cuas_multi_agent.agents[0].theta = np.pi
        
        
        self.cuas_multi_agent.obstacles[0].x = 20
        self.cuas_multi_agent.obstacles[0].y = 20
        self.cuas_multi_agent.obstacles[0].theta = 0

        self.cuas_multi_agent.render()
        
        actions = self.cuas_multi_agent.action_space.sample()
        for action in actions:
            action = self.cuas_multi_agent._unscale_action(action)

        actions[0] = [1, 0]
        des_v = self.cuas_multi_agent.uni_to_si_dyn(
                    self.cuas_multi_agent.agents[0], actions[0]
                )
        print(des_v)
        
        des_v = compute_velocity(self.cuas_multi_agent.agents[0], self.cuas_multi_agent.agents, self.cuas_multi_agent.obstacles, des_v, 10, 1)

        new_action = self.cuas_multi_agent.si_uni_dyn(
                    self.cuas_multi_agent.agents[0], des_v
                )

    def test_transformations(self):
        self.cuas_multi_agent.reset()

        actions = self.cuas_multi_agent.action_space.sample()

        for i in range(20):
            for i, action in actions.items():
                if self.cuas_multi_agent.agents[i].type == AgentType.E:
                    continue
                action = self.cuas_multi_agent._unscale_action(action)

                des_v = self.cuas_multi_agent.uni_to_si_dyn(
                    self.cuas_multi_agent.agents[i], action
                )
                
                # des_v = compute_velocity(self.cuas_multi_agent.agents[i], self.cuas_multi_agent.agents, self.cuas_multi_agent.obstacles, des_v, 10, 1)

                new_action = self.cuas_multi_agent.si_uni_dyn(
                    self.cuas_multi_agent.agents[i], des_v
                )

                np.testing.assert_almost_equal(action, new_action)

    @unittest.skip
    def test_compute_velocity(self):
        self.cuas_multi_agent.reset()

        total_rewards = {i: 0 for i in range(self.cuas_multi_agent.num_agents)}

        target_collisions = {i: 0 for i in range(self.cuas_multi_agent.num_agents)}
        agent_collisions = {i: 0 for i in range(self.cuas_multi_agent.num_agents)}
        obstacle_collisions = {i: 0 for i in range(self.cuas_multi_agent.num_agents)}
        for i in range(1500):
            # actions = {
            #     i: np.array([1, 0]) for i in range(self.cuas_multi_agent.num_agents)
            # }

            actions = self.cuas_multi_agent.action_space.sample()
            obs, reward, done, info = self.cuas_multi_agent.step(actions)
            self.cuas_multi_agent.render()
            total_rewards = {i: total_rewards[i] + val for i, val in reward.items()}

            target_collisions = {
                i: target_collisions[i] + info[i]["target_collision"]
                for i in info.keys()
            }
            agent_collisions = {
                i: agent_collisions[i] + info[i]["agent_collision"] for i in info.keys()
            }
            obstacle_collisions = {
                i: obstacle_collisions[i] + info[i]["obstacle_collision"]
                for i in info.keys()
            }

            if done["__all__"]:
                obs = self.cuas_multi_agent.reset()
        print(
            f"total_rewards: {total_rewards}\ntarget_collisions: {target_collisions}\nagent_collisions: {agent_collisions}\nobstacle_collisions: {obstacle_collisions}"
        )
        self.assertFalse(False)
        
        self.cuas_multi_agent.close()


if __name__ == "__main__":
    unittest.main()
