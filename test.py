import numpy as np
import random
import matplotlib.pyplot as plt

import matrix_op
from flight import FlightStatus, Airplane, Environment

def test_rotation_around_arbitrary_axis():
    v = random_vect_4d(5) # vector direction around which the rotation happens
    p = random_vect_4d(5) # a point on the vector
    some_point = random_vect_4d(5) # point to be rotated

    x, y, z = [], [], []
    for alpha in np.linspace(0, np.pi, 10):
        R = matrix_op.rotate_arb_axis(
            p=p,
            v=v,
            alpha=alpha
        )
        new_point = R.dot(some_point)
        x.append(new_point[0][0])
        y.append(new_point[1][0])
        z.append(new_point[2][0])

    # plot the rotation and the vector
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(p[0][0], p[1][0], p[2][0], v[0][0], v[1][0], v[2][0], normalize=False, color='brown')
    ax.scatter(x, y, z, color='blue', marker='o')
    ax.scatter([some_point[0][0]], [some_point[1][0]], [some_point[2][0]], color='red', marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # draw a cube large enough to hopefully make the axes equally scaled
    r = [-8, 8]
    from itertools import product, combinations
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="b")

    plt.show()

def random_vect_4d(scale):
    vect = []
    for _ in range(3):
        vect.append(random.uniform(-scale, scale))
    vect.append(1)
    return np.transpose(np.array([vect]))

def test_left_vectors_and_up():
    '''Shows the forward vector and its associated left vectors for different values of roll'''
    forward_vect = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
    forward_vect = forward_vect / np.linalg.norm(forward_vect)
    state = FlightStatus(forward_vect=forward_vect)
    x, y, z = [], [], []
    for roll in np.linspace(0, np.pi, 8):
        # roll = 0
        rotation = matrix_op.rotate_arb_axis(np.zeros((3,)), state.forward_vect, roll)
        left_vect = rotation.dot(matrix_op.vect_3dto4d(state.left_vect_no_roll))
        x.append(left_vect[0])
        y.append(left_vect[1])
        z.append(left_vect[2])
        
    # plot forward vector and left vectors
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(0, 0, 0, state.forward_vect[0], state.forward_vect[1], state.forward_vect[2], normalize=False, color='brown')
    ax.scatter(x, y, z, color='blue', marker='o')
    ax.scatter([state.left_vect_no_roll[0]], [state.left_vect_no_roll[1]], [state.left_vect_no_roll[2]], color='red', marker='^')
    ax.scatter([state.up_vect[0]], [state.up_vect[1]], [state.up_vect[2]], color='green', marker='^')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # draw a cube large enough to hopefully make the axes equally scaled
    r = [-3, 3]
    from itertools import product, combinations
    for s, e in combinations(np.array(list(product(r,r,r))), 2):
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s,e), color="b")

    plt.show()
    
def test_lift_drag():
    status = FlightStatus(velocity=np.array([250,0,0]), weight=np.array([0,0,-9.81*168000]))
    env = Environment()
    aircraft = Airplane(status, env, 418)
    for pitch in np.linspace(-np.pi/10, np.pi/5, 20):
        aircraft.status.forward_vect = np.array([1,0,np.tan(pitch)])
        aircraft.status.forward_vect = aircraft.status.forward_vect / np.linalg.norm(aircraft.status.forward_vect)
        aircraft.status.thrust = aircraft.status.forward_vect * 9.81 * 42637.683
        print(aircraft.status_str)

        lift = aircraft.lift
        drag = aircraft.drag
        thrust = aircraft.status.thrust
        weight = aircraft.status.weight

        # plot forces
        fig = plt.figure()
        f_plot = fig.add_subplot(111, projection='3d')

        vects = [lift, drag, thrust, weight]
        colors = ['blue', 'red', 'green', 'black']
        max_norm = 0
        for vect, color in zip(vects, colors):
            max_norm = max(max_norm, np.linalg.norm(vect))
            f_plot.quiver(0, 0, 0, vect[0], vect[1], vect[2], normalize=False, color=color)
        f_plot.set_xlabel('x')
        f_plot.set_ylabel('y')
        f_plot.set_zlabel('z')

        # draw a cube large enough to hopefully make the axes equally scaled
        r = [-max_norm, max_norm]
        from itertools import product, combinations
        for s, e in combinations(np.array(list(product(r,r,r))), 2):
            if np.sum(np.abs(s-e)) == r[1]-r[0]:
                f_plot.plot3D(*zip(s,e), color="b")

        plt.show()

if __name__ == '__main__':
    test_lift_drag()