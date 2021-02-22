import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.animation as animation
import matplotlib

def plot_scatter_animation_3d(data, dt):
    '''
    data: a list of lists of vectors. 
    Each element of the parent list contains the list of points (vectors) at a particular time t. 
    e.g. data[0] = list of points at t_0.
    e.g. data[0][0] = point_0 at t_0
    e.g. data[n][i] = point_i at t_n
    [
        [
            np.ndarray,
            np.ndarray,
            ...
        ],
        [
            np.ndarray,
            np.ndarray,
            ...
        ],
        [
            np.ndarray,
            np.ndarray,
            ...
        ],
        ...
    ]
    '''
    nfr = len(data) # Number of frames
    xs, ys, zs = [], [], []
    max_value = 0
    for data_for_time in data:
        x, y, z = [], [], []
        for point in data_for_time:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
            max_value = max([max_value, *[np.abs(coord) for coord in point]])
        xs.append(x)
        ys.append(y)
        zs.append(z)
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    print(xs, ys, zs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sct, = ax.plot([], [], [], "o", markersize=2)
    def update(n, xa, ya, za):
        sct.set_data(xa[n], ya[n])
        sct.set_3d_properties(za[n])
    ax.set_xlim(-max_value, max_value)
    ax.set_ylim(-max_value, max_value)
    ax.set_zlim(-max_value, max_value)
    ani = animation.FuncAnimation(fig, update, nfr, fargs=(xs, ys, zs), interval=1000*dt)
    
    plt.show()

if __name__ == '__main__':
    data = [
        [
            np.array([1,1,1]),
            np.array([-1,-1,-1]),
            np.array([-2,-2,-2]),
        ],
        [
            np.array([2,2,2]),
            np.array([-1,-1,-1]),
            np.array([0,0,0]),
        ]
    ]
    plot_scatter_animation_3d(data, 1)