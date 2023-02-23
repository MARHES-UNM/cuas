import context
import cuas
import unittest
from cuas.agents.cuas_agents import Agent
import numpy as np


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.pursuer = Agent(0, 0, 0)

    def test_agent(self):
        self.pursuer.step([10, 0])
        np.testing.assert_equal(np.array([0.1, 0, 0, 10, 0]), self.pursuer.state)

        self.pursuer.step([1, 0])
        np.testing.assert_equal(np.array([0.11, 0, 0, 1, 0]), self.pursuer.state)

        w = np.pi / 2 * 100
        self.pursuer.step([0, w])
        np.testing.assert_equal(
            np.array([0.11, 0, np.pi / 2, 0, w]), self.pursuer.state
        )

        self.pursuer.step([10, 0])
        np.testing.assert_equal(
            np.array([0.11, 0.1, np.pi / 2, 10, 0]), self.pursuer.state
        )

    def test_agent_rotation(self):
        w = np.pi * 100 + 10
        self.pursuer.step([0, w])
        np.testing.assert_almost_equal([0, 0, -np.pi + 0.1, 0, w], self.pursuer.state)

        self.pursuer.x, self.pursuer.y, self.pursuer.theta = 0, 0, 0
        w = 2 * np.pi * 100 + 10
        self.pursuer.step([0, w])
        np.testing.assert_almost_equal([0, 0, 0.1, 0, w], self.pursuer.state)

        self.pursuer.x, self.pursuer.y, self.pursuer.theta = 0, 0, 0
        w = -np.pi * 100 + -10
        self.pursuer.step([0, w])
        np.testing.assert_almost_equal([0, 0, np.pi + -0.1, 0, w], self.pursuer.state)

    def test_collision(self):
        pursuer = Agent(1, 0, 0, r=2.5)
        entity = Agent(1, 5, 0)
        self.assertFalse(pursuer.collision_with(entity))
        self.assertTrue(pursuer.collision_with(Agent(5, 1, 0, r=2.51)))


if __name__ == "__main__":
    unittest.main()