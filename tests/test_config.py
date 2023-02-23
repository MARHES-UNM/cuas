import unittest
import context
from cuas.utils.config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = {
            "a": 5,
            "b": {"c": 10},
            "d": "ok",
            "f": [1, 2, 3],
            "is_true": True,
            "is_false": False,
        }
        self.config = Config(self.cfg)

    def test_config(self):
        self.assertTrue(self.config.is_true)
        self.assertFalse(self.config.is_false)

        self.assertTrue(isinstance(self.config.f, list))

    def test_singleton(self):
        temp_config = Config(self.cfg)
        self.assertTrue(self.config is temp_config)


if __name__ == "__main__":
    unittest.main()
