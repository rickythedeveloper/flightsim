import numpy as np

import matrix_check

def rotate_x_4d(theta) -> np.ndarray:
    '''Returns a rotation matrix around x'''
    R = np.array(
        [[1, 0, 0, 0], 
        [0, np.cos(theta), -np.sin(theta), 0], 
        [0, np.sin(theta), np.cos(theta), 0], 
        [0, 0, 0, 1]]
    )
    return R

def rotate_y_4d(theta) -> np.ndarray:
    '''Returns a rotation matrix around y'''
    R = np.array(
        [[np.cos(theta), 0, np.sin(theta), 0], 
        [0, 1, 0, 0], 
        [-np.sin(theta), 0, np.cos(theta), 0], 
        [0, 0, 0, 1]]
    )
    return R

def rotate_z_4d(theta) -> np.ndarray:
    '''Returns a rotation matrix around z'''
    R = np.array(
        [[np.cos(theta), -np.sin(theta), 0, 0], 
        [np.sin(theta), np.cos(theta), 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]]
    )
    return R

def translate_4d(translation: np.ndarray) -> np.ndarray:
    '''Returns a translation matrix'''
    if translation.shape != (4, 1):
        raise TypeError('wrong shape of translation vector: ', translation)
    dx = translation[0][0]
    dy = translation[1][0]
    dz = translation[2][0]
    T = np.array(
        [[1, 0, 0, dx], 
        [0, 1, 0, dy], 
        [0, 0, 1, dz], 
        [0, 0, 0, 1]]
    )
    return T

def add_vect_4d(v1, v2):
    '''Returns the addition of two vectors that are 4D (includes translation as the fourth element)'''
    addition = v1 + v2
    addition[3][0] = 1
    return addition

def rotate_arb_axis(p: np.ndarray, v: np.ndarray, alpha: float) -> np.ndarray:
    '''
    Returns the 4x4 matrix of rotation around an arbitrary axis by angle alpha.
    The arbitrary axis is specified by a point vector p from the origin to any point on the axis and the axis direction vector v.
    '''
    # Correct the shape of p and v if it's predictable
    p = vect_3dto4d(p)
    v = vect_3dto4d(v)
    assert p.shape == (4, 1) and v.shape == (4, 1), f'wrong shape in either p or v: p:{p.shape}, v:{v.shape}'
    if matrix_check.is_zero_vector(v):
        print(f'v should not be a zero vector when rotating around it. We will proceed by not rotating. v:{v}')
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    vx = v[0][0]
    vy = v[1][0]
    vz = v[2][0]
    # print(v)

    if vx == 0 and vy == 0:
        z_rot_angle = 0.0
        y_rot_angle = np.arctan(np.sign(vz) * np.infty) # vector aligned to +ve x direction
        x_rot_angle = alpha
    else:
        if vx == 0:
            z_rot_angle = -np.arctan(np.sign(vy) * np.infty) # vector in the +ve x side
            y_rot_angle = np.arctan(vz / np.sqrt(vx**2 + vy**2)) # vector aligned to +ve x direction
            x_rot_angle = alpha
        else:
            z_rot_angle = -np.arctan(vy/vx) # vector either +ve or -ve x side
            y_rot_angle = np.sign(vx) * np.arctan(vz / np.sqrt(vx**2 + vy**2)) # vector aligned to either +ve or -ve x direction
            x_rot_angle = np.sign(vx) * alpha

    axis_endpoint = add_vect_4d(p, v)

    # translate the vector onto x-z plane
    # print('translate')
    alignmentMat = translate_4d(-p)
    # print(alignmentMat.dot(axis_endpoint))
    assert matrix_check.is_on_origin(alignmentMat.dot(p)), f'the alignment matrix does not align p onto the origin\nalignment matrix:\n{alignmentMat},\np:\n{p},\nresult:\n{alignmentMat.dot(p)}'

    # rotate around z axis to align the vector onto the xz plane
    # print(f'rotate around z {z_rot_angle * 180 / np.pi}')
    alignmentMat = rotate_z_4d(z_rot_angle).dot(alignmentMat)
    # print(alignmentMat.dot(axis_endpoint))
    assert matrix_check.is_on_xz_plane(alignmentMat.dot(p)), 'p is not aligned on xz plane'
    assert matrix_check.is_on_xz_plane(alignmentMat.dot(axis_endpoint)), 'p+v is not aligned on xz plane'
    
    # rotate around y axis to align the vector onto x axis
    # print(f'rotate around y by {y_rot_angle * 180 / np.pi}')
    alignmentMat = rotate_y_4d(y_rot_angle).dot(alignmentMat)
    # print(alignmentMat.dot(axis_endpoint))
    assert matrix_check.is_on_origin(alignmentMat.dot(p)), f'the alignment matrix does not align p onto the origin\nalignment matrix:\n{alignmentMat},\np:\n{p},\nresult:\n{alignmentMat.dot(p)}'
    assert matrix_check.is_on_xaxis(alignmentMat.dot(axis_endpoint)), f'the alignment matrix does not align p+v onto x axis \nalignmentMat\n. {axis_endpoint}\n= {alignmentMat.dot(axis_endpoint)}'

    # rotate around x axis as required by alpha or -alpha
    # print(f'rotate around x by {x_rot_angle}')
    rotationMat = rotate_x_4d(x_rot_angle)

    # reverse procedure
    # this matrix should be the inverse of alignmentMat
    # print('reverse the alignment matrix')
    reverseMat = translate_4d(p).dot(rotate_z_4d(-z_rot_angle).dot(rotate_y_4d(-y_rot_angle)))
    assert matrix_check.is_indentity(reverseMat.dot(alignmentMat)), 'recoveryMat dot alignmentMat is not identity'

    # overall transformation matrix
    R = reverseMat.dot(rotationMat.dot(alignmentMat))
    assert matrix_check.are_equal_vectors(p, R.dot(p)), 'p does not come back to the same point after the rotation op'
    assert matrix_check.are_equal_vectors(axis_endpoint, R.dot(axis_endpoint)), 'p+v does not come back to the same point after the rotation op'

    return R

def vect_3dto4d(v):
    assert v.shape == (3, 1) or v.shape == (3,), 'not the right shape'
    v = np.append(v, 1)
    v = np.array([v]).T
    return v

def vect_4dto3d(v):
    assert v.shape == (4,1), 'not the right shape'
    v = v[:3]
    assert v.shape == (3,1)
    return v

def vect_column_2_row(v):
    assert v.shape == (3,1) or v.shape == (4,1), f'shape is wrong: {v.shape}'
    v = v.T
    new_v = np.array([])
    for entry in v[0]:
        new_v = np.append(new_v, entry)
    assert new_v.shape == (3,) or new_v.shape == (4,)
    return new_v