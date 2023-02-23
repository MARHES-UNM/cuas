import math
import unittest

import cuas
import gym
import numpy as np
from cuas.envs import cuas_env_multi_agent
from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1
from cuas.models.constraint_model import ConstraintModel
from cuas.utils.replay_buffer import ReplayBuffer

import context


class TestUtil(unittest.TestCase):
    def setUp(self):
        self._replay_buffer = ReplayBuffer(5)
        config = {"num_obstacles": 3, "num_pursuers": 4, "seed": 40}
        self.cuas_multi_agent = CuasEnvMultiAgentV1(config)

    def test_buffer(self):
        self.assertEqual(self._replay_buffer._buffer_size, 5)

        actions = self.cuas_multi_agent.action_space.sample()

        obs = self.cuas_multi_agent.reset()
        
        

        constraints = self.cuas_multi_agent.get_constraints()

        obs_next, _, done, _ = self.cuas_multi_agent.step(actions)

        constraints_next = self.cuas_multi_agent.get_constraints()

        for (_, action), (_, ob), (_, c), (_, c_next) in zip(
            actions.items(), obs.items(), constraints.items(), constraints_next.items()
        ):
            print(action)
            print(ob)
            print(c)

            self._replay_buffer.add(
                {"action": action, "observation": ob, "c": c, "c_next": c_next}
            )

        self.assertEqual(
            self._replay_buffer._current_index, self.cuas_multi_agent.num_agents
        )


    def test_constraint(self):
        obs_space = gym.spaces.Box(
            low=self.cuas_multi_agent.norm_low_obs, high=self.cuas_multi_agent.norm_high_obs, dtype=np.float32
        )
        act_space = gym.spaces.Box(
            low=self.cuas_multi_agent.norm_low, high=self.cuas_multi_agent.norm_high, dtype=np.float32
        )
        
        c_model = ConstraintModel(self.cuas_multi_agent.agent_num_states, act_space.shape[0])
        
        
        

if __name__ == "__main__":
    unittest.main()
