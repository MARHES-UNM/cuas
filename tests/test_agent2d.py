import context
import cuas
import unittest
from cuas.agents.cuas_agents import Agent2D
import numpy as np


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.pursuer = Agent2D(0, 0, 0)

    def test_agent2d(self):
        self.pursuer.step([10, 0])
        np.testing.assert_equal(np.array([0.1, 0, 0, 10, 0]), self.pursuer.state)

        self.pursuer.step([1, 0])
        np.testing.assert_equal(np.array([0.11, 0, 0, 1, 0]), self.pursuer.state)

        self.pursuer.step([5, 5])
        np.testing.assert_equal(
            np.array([0.16, 0.05, np.pi / 4, 5, 5]), self.pursuer.state
        )

    def test_agent_min_max(self):
        self.pursuer.step([8000, 6000])
        np.testing.assert_almost_equal(
            np.array([80, 60, np.arctan2(6000, 8000), 8000, 6000]), self.pursuer.state
        )

    def test_collision(self):
        pursuer = Agent2D(1, 0, 0, r=2.5)
        entity = Agent2D(1, 5, 0)
        self.assertFalse(pursuer.collision_with(entity))
        self.assertTrue(pursuer.collision_with(Agent2D(5, 1, 0, r=2.51)))


if __name__ == "__main__":
    unittest.main()