import numpy as np
from typing import List

import matrix_check
import matrix_op

class PointMass():
    def __init__(self, mass: float, position: np.ndarray):
        self.mass = mass
        self.position = position.astype(float)

    def __str__(self):
        return f'{self.mass}kg at {self.position}'

    def __repr__(self):
        return str(self)

    def ang_mom_inertia(self, axis_origin: np.ndarray, axis_direction: np.ndarray):
        '''Returns the angular moment of inertia given the position / direction of the axis around which we measure the moment.'''
        mass_relative_pos = self.position - axis_origin
        axis_to_mass = mass_relative_pos - axis_direction * np.dot(axis_direction, mass_relative_pos)
        dis_axis_to_mass = np.linalg.norm(axis_to_mass)
        mom_inertia = self.mass * dis_axis_to_mass**2
        return mom_inertia

class PhysicsBody():
    def __init__(self, masses, velocity, ang_velocity):
        self.masses: List[PointMass] = masses
        self.velocity = velocity.astype(float)
        self.ang_velocity = ang_velocity.astype(float)

    @property
    def total_mass(self):
        return sum([mass.mass for mass in self.masses])

    @property
    def centre_mass(self):
        mass_pos_prod = sum([mass.mass * mass.position for mass in self.masses])
        centre = mass_pos_prod / self.total_mass
        assert type(centre) == np.ndarray
        return centre

    def apply_impulse(self, I, position):
        '''
        Updates the velocity and angular velocity of the body given a force and a time step.
        I: 3D vector of the impulse (= force x time) applied
        position: 3D position vector of the position where the force is applied.
        '''
        # record current angular momentum
        old_ang_momentum = self.ang_mom_inertia(self.center_mass, self.ang_velocity) * self.ang_velocity

        # update velocity
        self.velocity += I / self.total_mass
        
        # update angular velocity
        r = position - self.centre_mass
        delta_ang_momentum = np.cross(r, I) # this is effectively torque x dt
        new_ang_momentum = old_ang_momentum + delta_ang_momentum
        self.ang_velocity = new_ang_momentum / self.ang_mom_inertia(self.centre_mass, new_ang_momentum)

    def time_march(self, dt):
        old_ang_inertia = self.ang_mom_inertia(self.centre_mass, self.ang_velocity)
        old_ang_vel = self.ang_velocity

        # update position
        centre_mass = self.centre_mass # store the current centre of mass
        for point_mass in self.masses:
            delta_position_translation = self.velocity * dt
            rotation = matrix_op.rotate_arb_axis(centre_mass, self.ang_velocity, np.linalg.norm(self.ang_velocity) * dt)
            position_4d = matrix_op.vect_3dto4d(point_mass.position)
            new_position_4d = rotation.dot(position_4d)
            point_mass.position = matrix_op.vect_column_2_row(matrix_op.vect_4dto3d(new_position_4d)) + delta_position_translation
        
        # update angular velocity
        new_ang_inertia = self.ang_mom_inertia(self.centre_mass, self.ang_velocity)
        self.ang_velocity = (old_ang_inertia * self.ang_velocity) / new_ang_inertia # this conserves the angular momentum

        # print(f'centre mass\n{self.centre_mass}\nbodies\n{self.masses}\n')
        # if not matrix_check.are_equal_vectors(old_ang_vel, self.ang_velocity):
            # print(f'angular velocity changed during time march from {old_ang_vel} to {self.ang_velocity}')

        assert matrix_check.are_equal_vectors(old_ang_inertia * old_ang_vel, new_ang_inertia * self.ang_velocity), f'angular momentum changed from {old_ang_inertia * old_ang_vel} to {new_ang_inertia * self.ang_velocity} but it shouldnt change during a time march'

    def ang_mom_inertia(self, axis_origin: np.ndarray, axis_direction: np.ndarray):
        return sum([mass.ang_mom_inertia(axis_origin, axis_direction) for mass in self.masses])

def uniform_masses(mass, body_equation, min_coord: np.ndarray, max_coord: np.ndarray, step):
    '''
    Returns a list of point masses that represents a body mass given its information. The arguments are:
    mass: mass of the box in kg.
    body_equation: a callable function which accepts a position vector as an input and returns a bool indicating whether that point is in the body.
    min_coord: a 3D position vector containing minx, miny, minz from which we should vetify whether the body covers particular points.
    max_coord: a 3D position vector containing maxx, maxy, maxz at which we should stop verifying whether the body covers particular points.
    step: the rough length between the point masses that approximate this body of mass.
    '''
    min_x, min_y, min_z = min_coord
    max_x, max_y, max_z = max_coord
    point_masses = []
    for x in np.linspace(min_x, max_x, int((max_x - min_x)/step)+1):
        for y in np.linspace(min_y, max_y, int((max_y - min_y)/step)+1):
            for z in np.linspace(min_z, max_z, int((max_z - min_z)/step)+1):
                pos = np.array([x,y,z])
                if body_equation(pos):
                    point_masses.append(PointMass(0, pos))

    each_mass = mass / len(point_masses)
    for point_mass in point_masses:
        point_mass.mass = each_mass

    return point_masses