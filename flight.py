import numpy as np

import matrix_op
import matrix_check

class FlightStatus():
    def __init__(self, forward_vect=None, roll=0.0, velocity=None, position=None, thrust=None, mass=1000):
        '''
        direction: 3D np unit vector of the nose direction
        roll: float indicating the roll (+ve: R, -ve: L)
        velocity: 3D np vector containing x,y,z speeds in m/s
        position: 3D np vector containing x,y,z positions m
        thrust: 3D np vector of the thrust force
        mass: mass of the flying object in kg
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
        self.mass = mass

    def __str__(self):
        return f'forward: {self.forward_vect}, roll: {self.roll}, velocity: {self.velocity}, thrust: {self.thrust}, weight: {self.weight}, position: {self.position}'

    @property
    def weight(self):
        g = 9.81
        return np.array([0,0,-self.mass*g])

    @property
    def roll(self):
        return self._roll

    @roll.setter
    def roll(self, angle):
        self._left_vect = None
        self._roll = angle

    @property
    def forward_vect(self):
        return self._forward_vect

    @forward_vect.setter
    def forward_vect(self, vect):
        norm = np.linalg.norm(vect)
        if not matrix_check.is_normalised(vect):
            print(f'forward_vect is not normalised (norm={norm}). But we will normalise it and proceed.')
        self._forward_vect = vect / norm

        # remove all cached data dependent on forward vect
        self._left_vect_no_roll = None
        self._left_vect = None

    @property
    def left_vect_no_roll(self):
        '''The left vector of the object if the roll were zero'''
        # if cached, return immediately
        if self._left_vect_no_roll is not None:
            return self._left_vect_no_roll

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
        self._left_vect_no_roll = left_vect_roll_zero
        return left_vect_roll_zero

    @property
    def left_vect(self):
        '''The left vector of the object.'''
        # if cached, return immediately
        if self._left_vect is not None:
            return self._left_vect
        
        origin = np.array([0, 0, 0]).T
        current_left_vect = matrix_op.rotate_arb_axis(origin, self.forward_vect, self.roll).dot(matrix_op.vect_3dto4d(self.left_vect_no_roll))
        current_left_vect = matrix_op.vect_column_2_row(matrix_op.vect_4dto3d(current_left_vect))
        current_left_vect = current_left_vect / np.linalg.norm(current_left_vect)
        assert matrix_check.is_normalised(current_left_vect), f'left vect is not normalised. This is an internal problem. norm={np.linalg.norm.norm(current_left_vect)}'
        assert matrix_check.are_perp(current_left_vect, self.forward_vect), f'left_vect dot forward_vect is non-zero (= {np.dot(current_left_vect, self.forward_vect)})'
        self._left_vect = current_left_vect
        return current_left_vect

    @property
    def up_vect(self):
        return np.cross(self.forward_vect, self.left_vect)

class Environment():
    def __init__(self):
        return

    def wind(self, position):
        return np.array([1,2,0])

    def air_density(self, position):
        return 0.4135

class FlyingObject():
    def __init__(self, status: FlightStatus, env: Environment):
        self.status = status
        self.env = env

    @property
    def net_force(self):
        raise NotImplementedError('The net force calculation has to be implemented.')

    @property
    def acceleration(self):
        return self.net_force / self.status.mass

    
class Airplane(FlyingObject):
    def __init__(self, status: FlightStatus, env: Environment, wing_area):
        super().__init__(status, env)
        self.wing_area = wing_area

    @classmethod
    def default_airplane(cls):
        forward_vect = np.array([1.0,0.0,0.2])
        roll = 0
        velocity = np.array([200.0, 0.0, -10])
        position = np.array([0.0, 0.0, 10000.0])
        thrust = forward_vect * 330000.0
        mass = 168000.0
        status = FlightStatus(forward_vect, roll, velocity, position, thrust, mass)
        env = Environment()
        wing_area = 420.0
        plane = Airplane(status, env, wing_area)
        return plane

    @property
    def status_str(self):
        return f'{self.status}\nlift: {self.lift}, lift coeff: {self.lift_coeff}\ndrag: {self.drag}, drag coeff: {self.drag_coeff}'

    @property
    def drag_coeff(self):
        '''Returns the drag coefficient for angle of attack alpha'''
        return 0.10

    @property
    def drag(self):
        '''3D vector of the drag force in N'''
        air_v = self.air_velocity
        rho = self.env.air_density(self.status.position)
        mag = self.drag_coeff * 0.5 * rho * np.linalg.norm(air_v)**2 * self.wing_area
        unit_vect = - air_v / np.linalg.norm(air_v)
        return mag * unit_vect

    @property
    def lift_coeff(self):
        alpha = self.angle_attack
        alpha_deg = alpha * 180 / np.pi
        if alpha_deg < 15:
            gradient = 1.50 / 20
            coeff = gradient * (alpha_deg - (-5))
        else:
            gradient = -1.50 / 20
            coeff = 1.75 + gradient * (alpha_deg - 15)

        coeff = max(coeff, 0)
        return coeff

    @property
    def lift(self):
        '''3D vector of the lift force in N'''
        air_v = self.air_velocity
        rho = self.env.air_density(self.status.position)
        mag = self.lift_coeff * 0.5 * rho * np.linalg.norm(air_v)**2 * self.wing_area
        direction = np.cross(air_v, self.status.left_vect)
        unit_vect = direction / np.linalg.norm(direction)
        return mag * unit_vect

    @property
    def net_force(self):
        f = self.status.thrust + self.status.weight + self.lift + self.drag
        return f

    @property
    def angle_attack(self):
        vz = np.dot(self.status.up_vect, self.air_velocity)
        vx = np.dot(self.status.forward_vect, self.air_velocity)
        alpha = - np.arctan(vz/vx)
        return alpha

    @property
    def air_velocity(self):
        wind = self.env.wind(self.status.position)
        return self.status.velocity - wind

    @property
    def velocity_pitch(self):
        horizontal_v = np.sqrt(self.status.velocity[0]**2 + self.status.velocity[1]**2)
        vertical_v = self.status.velocity[2]
        pitch = np.arctan(vertical_v / horizontal_v)
        return pitch