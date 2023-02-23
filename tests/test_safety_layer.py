from asyncio import constants
import context
import math
import unittest

import cuas
import gym
import numpy as np
from cuas.envs.cuas_env_multi_agent_v1 import CuasEnvMultiAgentV1
from cuas.models.constraint_model import ConstraintModel
from cuas.utils.replay_buffer import ReplayBuffer
from cuas.safety_layer.safety_layer import SafetyLayer


class TestSafetyLayer(unittest.TestCase):
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
            "constraint_slack": 1.5,
            "pursuer_drive_type": "CuasAgent2D",
        }
        sl_config = {
            # "replay_buffer_size": 1000, "episode_length": 200
            # "num_epochs": 200,
        }
        self.env = CuasEnvMultiAgentV1(env_config)
        self._safety_layer = SafetyLayer(self.env, sl_config)

    # @unittest.skip
    def test_sample_step(self):
        self._safety_layer._sample_steps(5)

    @unittest.skip
    def test_train(self):
        self._safety_layer = SafetyLayer(
            self.env, {"num_epochs": 5, "sl_lr": 5e-4}
        )
        self._safety_layer.train()

    @unittest.skip
    def test_layer(self):
        self.eval_agent(use_safe_layer=False)
        self.eval_agent(use_safe_layer=True)

    @unittest.skip
    def test_use_safety_action(self):
        self.env.use_safety_layer = True
        self.env.safety_layer_type = "hard"
        for step in range(20):

            done = {"__all__": False}
            obs = self.env.reset()
            while not done["__all__"]:

                actions = self.env.action_space.sample()
                # actions = {i: [1, .1] for i in range(self.env.num_agents)}
                obs_next, _, done, info = self.env.step(actions)

                self.env.render()

    def eval_agent(self, use_safe_layer=False):

        if use_safe_layer:
            self._safety_layer.load_layer()

        obs = self.env.reset()

        total_rewards = {i: 0 for i in range(self.env.num_agents)}
        target_collisions = {i: 0 for i in range(self.env.num_agents)}
        agent_collisions = {i: 0 for i in range(self.env.num_agents)}
        obstacle_collisions = {i: 0 for i in range(self.env.num_agents)}

        cum_target_collisions = 0
        cum_agent_collision = 0
        cum_obs_collisions = 0

        # constraints = {i:-10 * np.ones(9) for i in range(self.env.num_agents)}

        for step in range(20):

            done = {"__all__": False}
            obs = self.env.reset()
            constraints = self.env.get_constraints()
            while not done["__all__"]:

                actions = self.env.action_space.sample()
                # for k, v in actions.items():
                #     actions[k] = self.env.agents[k].uni_to_si_dyn(v)

                # constraints.self.env_get
                c_margin = self.env.get_constraint_margin()
                # actions = np.concatenate([v for k, v in sorted(actions.items())], 0)
                # constraints = np.concatenate(
                #     [v for k, v in sorted(constraints.items())], 0
                # )
                # obs = np.concatenate([v for k, v in sorted(obs.items())], 0)

                if use_safe_layer:
                    # new_actions = {}
                    # for (i, action), (_, ob), (_, c) in zip(
                    #     actions.items(),
                    #     obs.items(),
                    #     constraints.items(),
                    # ):
                    #     new_actions[i] = self._safety_layer.get_hard_safe_action(
                    #         ob, action, c
                    #     )
                    new_actions = self._safety_layer.get_hard_safe_action(
                        obs, actions, constraints
                    )

                    # for k, v in new_actions.items():
                    #     new_actions[k] = self.env.agents[k].si_to_uni_dyn(v)
                    #     new_actions[k] = np.clip(
                    #         new_actions[k], -1, 1
                    #     )

                    actions = new_actions

                obs_next, _, done, info = self.env.step(actions)

                obs = obs_next
                constraints = self.env.get_constraints()
                self.env.render()

                target_collisions = {
                    i: target_collisions[i] + info[i]["target_collision"]
                    for i in info.keys()
                }
                agent_collisions = {
                    i: agent_collisions[i] + info[i]["agent_collision"]
                    for i in info.keys()
                }
                obstacle_collisions = {
                    i: obstacle_collisions[i] + info[i]["obstacle_collision"]
                    for i in info.keys()
                }

        print(f"num_step: {step}")
        print(f"num_target_collision: {target_collisions}")
        print(f"num_agent_collisions: {agent_collisions}")
        print(f"num_obstacle_collisions: {obstacle_collisions}\n")

        for i in range(self.env.num_agents):
            cum_target_collisions += target_collisions[i]
            cum_agent_collision += agent_collisions[i]
            cum_obs_collisions += obstacle_collisions[i]

        print(f"cum target collisions: {cum_target_collisions}")
        print(f"cum agents collisions: {cum_agent_collision}")
        print(f"cum obstacle collisions: {cum_obs_collisions}")

    def test_arrays(self):
        c = self.env.get_constraints()

        # c = np.zeros([4, 9])
        ca = [v for k, v in c.items()]


if __name__ == "__main__":
    unittest.main()
