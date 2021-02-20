import unittest
import numpy as np

from matrix_check import are_equal, are_equal_vectors
from mass_physics import PointMass, PhysicsBody

class TestPhysics(unittest.TestCase):
    # def setUp(self):
    #     pass

    def test_constant_ang_velocity(self):
        masses = [
            PointMass(1.0, np.array([-1, -121, 12])),
            PointMass(0.5, np.array([-1, 124, -1286])),
        ]
        velocity = np.array([10,3,5])
        ang_velocity = np.array([1,0,0])
        body = PhysicsBody(masses, velocity, ang_velocity)
        dt = 0.01
        t_final = 1000
        for _ in range(int(t_final/dt)):
            body.time_march(dt)

        self.assertTrue(are_equal_vectors(ang_velocity, body.ang_velocity))



if __name__ == '__main__':
    unittest.main()