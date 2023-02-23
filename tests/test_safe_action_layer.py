from asyncio import constants
import context
import math
import unittest

import cuas
import gym
import numpy as np
from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1
from cuas.safety_layer.safe_action_layer import SafeActionLayer


class TestSafeActionLayer(unittest.TestCase):
    def setUp(self):
        env_config = {
            "obstacle_penalty": True,
            "agent_collision_penalty": True,
            "agent_radius": 1,
            "obstacle_radius": 4,
            "obstacle_v": 5,
            "render_trace": False,
            "seed": 123,
            "alpha": 0,
            "beta": 0.20,
            "evader_move_type": "repulsive",
            "evader_alpha": 1,
            "pursuer_move_type": "rl",
            "agent_penalty_weight": 0.1,
            "obstacle_penalty_weight": 0.15,
            "observation_type": "global",
            # "use_safety_layer": True,
            "num_pursuers": 4,
            "num_evaders": 1,
            "num_obstacles": 5,
            "constraint_slack": 1,
        }
        sl_config = {
            "replay_buffer_size": 1000, "episode_length": 200, "num_epochs": 2, 
        }
        self.env = CuasEnvMultiAgentV1(env_config)
        self._safe_action_layer = SafeActionLayer(self.env, sl_config)

    def test_sample_step(self):
        self._safe_action_layer._sample_steps(5)

    def test_train(self):
        self._safe_action_layer.train()

if __name__ == "__main__":
    unittest.main()
