import torch
import numpy as np
import math
from scipy.spatial.transform import Rotation
 
 

def RpToTrans(phix,phiy,phiz,px,py,pz):
    phix = phix[0] / 180 * math.pi
    phiy = phiy[0] / 180 * math.pi
    phiz = phiz[0] / 180 * math.pi
    px = px[0]
    py = py[0]
    pz = pz[0]
    R = Rotation.from_euler('xyz', [phix,phiy,phiz], degrees=True)
    R = R.as_matrix()
    p = np.array([[px],[py],[pz]])
    T = np.block([[R, p],[np.zeros((1, 3)), 1]])
    return T

def TransToRp(T):
    """Converts a homogeneous transformation matrix into a rotation matrix
    and position vector
    :param T: A homogeneous transformation matrix
    :return R: The corresponding rotation matrix,
    :return p: The corresponding position vector.
    Example Input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        (np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]]),
         np.array([0, 0, 3]))
    """
    T = np.array(T)
    R = T[0: 3, 0: 3]
    p = T[0: 3, 3]
    return R, p

def TransInv(T):
    """Inverts a homogeneous transformation matrix
    :param T: A homogeneous transformation matrix
    :return: The inverse of T
    Uses the structure of transformation matrices to avoid taking a matrix
    inverse, for efficiency.
    Example input:
        T = np.array([[1, 0,  0, 0],
                      [0, 0, -1, 0],
                      [0, 1,  0, 3],
                      [0, 0,  0, 1]])
    Output:
        np.array([[1,  0, 0,  0],
                  [0,  0, 1, -3],
                  [0, -1, 0,  0],
                  [0,  0, 0,  1]])
    """
    R, p = TransToRp(T)
    Rt = np.array(R).T
    TInv = np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

    return TInv



def se3ToVec(se3mat):
    """ Converts an se3 matrix into a spatial velocity vector
    :param se3mat: A 4x4 matrix in se3
    :return: The spatial velocity 6-vector corresponding to se3mat
    Example Input:
        se3mat = np.array([[ 0, -3,  2, 4],
                           [ 3,  0, -1, 5],
                           [-2,  1,  0, 6],
                           [ 0,  0,  0, 0]])
    Output:
        np.array([1, 2, 3, 4, 5, 6])
    """
    V = np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]
    
    return V
 
