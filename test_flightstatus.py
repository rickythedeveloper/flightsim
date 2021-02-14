import unittest
import numpy as np
from flight import FlightStatus
from matrix_check import are_equal, are_equal_vectors

class TestFlightStatus(unittest.TestCase):
    def setUp(self):
        forward_vect = np.array([3,5,6])
        roll = np.pi/6
        velocity = np.array([250, 0, 0])
        position = np.array([131203, 2047242, 13493])
        thrust = forward_vect / np.linalg.norm(forward_vect) * 418275
        mass = 168000
        self.flightstatus = FlightStatus(forward_vect, roll, velocity, position, thrust, mass)

    def test_attitude_vectors(self):
        forward_vect = self.flightstatus.forward_vect
        left_vect = self.flightstatus.left_vect
        up_vect = self.flightstatus.up_vect
        # norm = 1
        self.assertTrue(are_equal(1, np.linalg.norm(forward_vect)))
        self.assertTrue(are_equal(1, np.linalg.norm(left_vect)))

        # perpendicular 
        self.assertTrue(are_equal(0, np.dot(forward_vect, left_vect)))
        self.assertTrue(are_equal(0, np.dot(forward_vect, up_vect)))
        self.assertTrue(are_equal(0, np.dot(up_vect, left_vect)))

    def test_angle_attack(self):
        self.flightstatus.forward_vect = np.array([1,0,0])
        self.flightstatus.roll = 0
        wind = np.array([0, 0, 250])
        air_velocity = self.flightstatus.air_velocity(wind)
        alpha = self.flightstatus.angle_attack(wind)
        self.assertTrue(are_equal_vectors(air_velocity, np.array([250, 0, -250])))
        self.assertEqual(np.pi/4, alpha)

    def test_aa(self):
        self.assertEqual(1,1)



if __name__ == '__main__':
    unittest.main()