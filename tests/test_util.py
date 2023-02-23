import unittest
# import context
import cuas.util
import math

class TestUtil(unittest.TestCase):
    
    def test_angle(self):
        angle = cuas.util.angle(point_2=(0, 0))
        self.assertEqual(angle, 0.0)

        angle = cuas.util.angle(point_2=(1,0))
        self.assertEqual(angle, 0.0)
        
        angle = cuas.util.angle(point_2=(0,1))
        self.assertEqual(angle, math.pi /2)
        angle = cuas.util.angle(point_2=(0,-1))
        self.assertEqual(angle, -math.pi /2)
        angle = cuas.util.angle(point_2=(1,-1))
        self.assertEqual(angle, -math.pi /4)

    def test_distance(self):
        dist = cuas.util.distance(point_2=(0,0))
        self.assertEqual(dist, 0)

        dist = cuas.util.distance(point_1=(4,0))
        self.assertEqual(dist, 4)
        
        dist = cuas.util.distance(point_1=(5,5), point_2=(9,9))
        self.assertEqual(dist, math.sqrt(32))
        

    def test_norm_data(self):
        max = 5
        min = 0
        
        res = cuas.util.norm_data(0, max, min)

        self.assertEqual(res, -1)

        res = cuas.util.norm_data(5, max, min)
        self.assertEqual(res, 1)
        
        res = cuas.util.norm_data(4, max, min)
        self.assertAlmostEqual(res, .6)
        
        
if __name__ == "__main__":
    unittest.main()