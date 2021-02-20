import numpy as np

def is_on_xz_plane(vector):
    y = vector[1][0]
    return are_equal(y, 0)

def is_on_origin(vector):
    for n in range(3):
        value = vector[n][0]
        if not are_equal(value, 0):
            return False
    return True
    
def is_on_xaxis(vector):
    for n in range(1, 3):
        value = vector[n][0]
        if not are_equal(value, 0):
            return False
    return True

def are_equal(x1, x2):
    return np.abs(x1-x2) < 10**(-9)

def are_equal_vectors(v1, v2):
    for n in range(3):
        if not are_equal(v1[n], v2[n]):
            return False
    return True

def are_equal_matrices(a1, a2):
    if a1.shape != a2.shape:
        return False
    shape = a1.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not are_equal(a1[i, j], a2[i, j]):
                return False
    return True

def is_indentity(matrix):
    for index, entry in np.ndenumerate(matrix):
        i = index[0]
        j = index[1]
        if i == j:
            if not are_equal(entry, 1):
                return False
        else:
            if not are_equal(entry, 0):
                return False
    return True

def is_all_zeros(matrix):
    for _, entry in np.ndenumerate(matrix):
        if not are_equal(entry, 0):
            return False
    return True

def is_zero_vector(v):
    assert v.shape == (3,1) or v.shape == (3,) or v.shape == (4,1) or v.shape == (4,), 'shape of the vector is not right'
    for n in range(3):
        if not are_equal(v[n], 0):
            return False
    return True

def is_normalised(v):
    return are_equal(np.linalg.norm(v), 1)

def are_perp(v1, v2):
    assert v1.shape == (3,) and v2.shape == (3,), f'shape is wrong. v1: {v1.shape}, v2:{v2.shape}'
    return are_equal(np.dot(v1, v2), 0)