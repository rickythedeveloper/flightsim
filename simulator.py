import time
import sys
import numpy as np
from flight import FlyingObject, FlightStatus

class Simulator():
    def __init__(self, flyingObject: FlyingObject):
        self.flyingObject = flyingObject

    def start(self, dt=0.01, t_max=100, print_status=False):
        assert dt < t_max
        last_time = time.time()
        for _ in range(int(t_max / dt)):
            current = time.time()
            wait_time = last_time + dt - current
            if wait_time > 0:
                time.sleep(wait_time)
            self.march(dt)
            if print_status:
                sys.stdout.flush()

                angle_attack = int(self.flyingObject.angle_attack * 180 / np.pi)
                v_pitch = int(self.flyingObject.velocity_pitch * 180 / np.pi)
                velocity = self.flyingObject.status.velocity.astype(int)
                accel = self.flyingObject.acceleration
                thrust = self.flyingObject.status.thrust.astype(int)
                weight = self.flyingObject.status.weight.astype(int)
                drag = self.flyingObject.drag.astype(int)
                lift = self.flyingObject.lift.astype(int)
                print(angle_attack, v_pitch, velocity, accel, thrust, weight, lift, drag, end='\r')

            last_time = current

    def predict(self, data_fields: list, dt=0.01, t_max=100):
        assert dt < t_max
        possible_fields = ['v', 'pos', 'forward', 'thrust', 'weight', 'lift', 'drag', 'net_force']
        
        data = {}
        for possible_field in possible_fields:
                if possible_field in data_fields:
                    data[possible_field] = np.array([])

        for _ in range(int(t_max / dt)):
            self.march(dt)

            data_dict = {
                'v': self.flyingObject.status.velocity,
                'pos': self.flyingObject.status.position,
                'forward': self.flyingObject.status.forward_vect,
                'thrust': self.flyingObject.status.thrust,
                'weight': self.flyingObject.status.weight,
                'lift': self.flyingObject.lift,
                'drag': self.flyingObject.drag,
                'net_force': self.flyingObject.net_force,
            }

            for possible_field in possible_fields:
                if possible_field in data_fields:
                    datum = data_dict[possible_field]
                    if len(data[possible_field]) == 0:
                        data[possible_field] = np.append(data[possible_field], datum)
                    else:
                        data[possible_field] = np.vstack((data[possible_field], datum))

        return data

    def march(self, dt):
        # update velocity
        a = self.flyingObject.acceleration
        dv = a * dt
        self.flyingObject.status.velocity += dv

        # update position
        dp = self.flyingObject.status.velocity * dt
        self.flyingObject.status.position += dp


