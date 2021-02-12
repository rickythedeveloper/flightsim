import numpy as np

import matrix_op
import matrix_check

class FlightState():
    def __init__(self, forward_vect=None, roll=0.0, velocity=None, position=None, thrust=None, weight=9.81*1000):
        '''
        direction: 3D np unit vector of the nose direction
        roll: float indicating the roll (+ve: R, -ve: L)
        velocity: 3D np vector containing x,y,z speeds in m/s
        position: 3D np vector containing x,y,z positions m
        thrust: 3D np vector of the thrust force
        weight: float that signifies the weight in N
        '''
        if forward_vect is None:
            forward_vect = np.array([1,0,0])
        if velocity is None:
            velocity = np.zeros((3,))
        if position is None:
            position = np.zeros((3,))
        if thrust is None:
            thrust = np.zeros((3,))
        
        self.forward_vect = forward_vect
        self.roll = roll
        self.velocity = velocity
        self.position = position
        self.thrust = thrust
        self.weight = weight
    @property
    def forward_vect(self):
        return self._forward_vect

    @forward_vect.setter
    def forward_vect(self, vect):
        norm = np.linalg.norm(vect)
        if not matrix_check.is_normalised(vect):
            print(f'forward_vect is not normalised (norm={norm}). But we will normalise it and proceed.')
        self._forward_vect = vect / norm

    @property
    def left_vect_no_roll(self):
        '''The left vector of the object if the roll were zero'''
        # to find two vectors that sit on the plane perpendicular to forward vecror,
        # we need two arbitrary vectors that are not parallel to the forward vector.
        assert not matrix_check.is_zero_vector_3d(self.forward_vect), 'the forward vector cannot be a zero vector'
        if self.forward_vect[0] == 0 and self.forward_vect[1]:
            x1 = np.array([1, 0, 0])
            x2 = np.array([0, 1, 0])
        elif self.forward_vect[1] == 0 and self.forward_vect[2]:
            x1 = np.array([0, 1, 0])
            x2 = np.array([0, 0, 1])
        elif self.forward_vect[2] == 0 and self.forward_vect[0]:
            x1 = np.array([0, 0, 1])
            x2 = np.array([1, 0, 0])
        else:
            x1 = np.array([1, 0, 0])
            x2 = np.array([0, 1, 0])

        # vectors perpendicular to the forward vector
        # print(self.forward_vect, x1, x2)
        p1, p2 = np.cross(self.forward_vect, x1), np.cross(self.forward_vect, x2)

        # combinations of the two can express any vector in that plane.
        # find the left vect if roll were 0
        z1, z2 = p1[2], p2[2]
        if z1 == 0:
            left_vect_roll_zero = p1
        else:
            ratio = - z2 / z1
            left_vect_roll_zero = ratio * p1 + p2
        # forward vecror X the left vector when the roll is zero should face somewhat upwards
        if np.cross(self.forward_vect, left_vect_roll_zero)[2] < 0:
            left_vect_roll_zero = -left_vect_roll_zero
        
        # normalise
        left_vect_roll_zero = left_vect_roll_zero / np.linalg.norm(left_vect_roll_zero)

        assert matrix_check.are_perp(left_vect_roll_zero, self.forward_vect), f'left_vect_roll_zero dot forward_vect is non-zero (= {np.dot(left_vect_roll_zero, self.forward_vect)})'
        return left_vect_roll_zero

    @property
    def left_vect(self):
        '''The left vector of the object.'''
        origin = np.array([0, 0, 0]).T
        current_left_vect = matrix_op.rotate_arb_axis(origin, self.forward_vect, self.roll).dot(matrix_op.vect_3dto4d(self.left_vect_no_roll))
        current_left_vect = matrix_op.vect_column_2_row(matrix_op.vect_4dto3d(current_left_vect))
        current_left_vect = current_left_vect / np.linalg.norm(current_left_vect)
        assert matrix_check.is_normalised(current_left_vect), f'left vect is not normalised. This is an internal problem. norm={np.linalg.norm.norm(current_left_vect)}'
        assert matrix_check.are_perp(current_left_vect, self.forward_vect), f'left_vect dot forward_vect is non-zero (= {np.dot(current_left_vect, self.forward_vect)})'
        return current_left_vect


    @property
    def up_vect(self):
        return np.cross(self.forward_vect, self.left_vect)

