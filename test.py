import numpy as np
import random
import matplotlib.pyplot as plt
import matrix_op

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

if __name__ == '__main__':
    for n in range(10):
        test_rotation_around_arbitrary_axis()