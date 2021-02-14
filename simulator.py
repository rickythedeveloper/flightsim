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

    def march(self, dt):
        # update velocity
        a = self.flyingObject.acceleration
        dv = a * dt
        self.flyingObject.status.velocity += dv

        # update position
        dp = self.flyingObject.status.velocity * dt
        self.flyingObject.status.position += dp


