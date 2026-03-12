import numpy as np
from typing import List, Tuple, Union

def build_matrices_notime(theta_cone: float, 
                          theta_tilt: float, 
                          theta_yaw: float
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Builds matrices that do not depend on time: from frame 1 to 2, and from frame 3 to 4. """
    a1 = np.array([[1,0,0], 
                   [0,np.cos(theta_yaw), np.sin(theta_yaw)], 
                   [0,-np.sin(theta_yaw), np.cos(theta_yaw)]])
    a2 = np.array([[np.cos(theta_tilt),0,-np.sin(theta_tilt)], 
                   [0,1,0], 
                   [np.sin(theta_tilt),0,np.cos(theta_tilt)]])
    a3 = np.array([[1,0,0], 
                   [0,1,0], 
                   [0,0,1]])
    a12 = np.matmul(np.matmul(a1,a2),a3)
    a34 = np.array([[np.cos(theta_cone), 0, -np.sin(theta_cone)], 
                    [0,1,0], 
                    [np.sin(theta_cone), 0,np.cos(theta_cone)]])
    return a12, a34
    
    
def build_matrix_a23(theta_blade: Union[float, np.ndarray]
                     )-> np.ndarray:
    """Builds transformation matrix from frame 2 to 3, depends on time through theta_blade"""
    cos_t = np.cos(theta_blade)
    sin_t = np.sin(theta_blade)
    
    # Stack matrices: shape (3, 3, N)
    return np.array([[cos_t, sin_t, np.zeros_like(cos_t)],
                     [-sin_t, cos_t, np.zeros_like(cos_t)],
                     [np.zeros_like(cos_t), np.zeros_like(cos_t), np.ones_like(cos_t)]])

def build_matrix_a14(theta_cone: float, 
                     theta_tilt: float,
                     theta_yaw: float, 
                     a23: np.ndarray
                     ) -> np.ndarray:
    """Builds transformation matrix from frame 1 to 4, depends on time through a23.
    
    Handles both single 3x3 matrix and batched 3x3xN array inputs.
    """
    a12, a34 = build_matrices_notime(theta_cone, theta_tilt, theta_yaw)
    
    a14 = np.dot(a34, np.dot(a23, a12))
    
    return a14

def get_position(r: Union[float, np.ndarray], 
                 a12: np.ndarray,
                 a14: np.ndarray,
                 H: float,
                 L: float
                ) -> np.ndarray:
    """Calculate the position in frame 1 of a point on the blade at distance r from the hub. 
    It also works for an array of distances r. """

    pre_rT = np.ones(len(r))*H   
    rT =  np.array([pre_rT, np.zeros_like(pre_rT), np.zeros_like(pre_rT)])
    a21 = a12.transpose()
    pre_rS = np.ones(len(r))*(-L)
    pre_rS2 = np.array([np.zeros_like(pre_rS), np.zeros_like(pre_rS), pre_rS])
    rS =np.dot( a21,pre_rS2)
    
    a41 = a14.transpose()
    rB = np.dot(a41, np.array([r, np.zeros_like(r), np.zeros_like(r)]))

    return rT + rS + rB