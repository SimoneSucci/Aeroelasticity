import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hipersim import MannTurbulenceField
from pathlib import Path
import sys


def get_pitch(time, switch1, switch2, pitch_value):
    if time<switch1 or time >switch2:
        theta_pitch = [0,0,0]
    else:
        theta_pitch = [pitch_value,pitch_value,pitch_value]
    return theta_pitch

def load_blade_data(txt_file: str
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Loads the blade data and records in an array for each characteristics"""

    blade_data = np.loadtxt(txt_file)
    radii = blade_data[:,0]
    chords = blade_data[:,2]
    betas = blade_data[:,1]
    thicknesses = blade_data[:,3]
    length = len(blade_data)

    return radii, chords, betas, thicknesses, length


def load_airfoils(thickness1_file: str, 
                  thickness2_file: str,
                  thickness3_file: str,
                  thickness4_file: str,
                  thickness5_file: str,
                  cylinder_file: str
                  )-> List:
    """Loads the airfoil data: CT and Cd for each airfoil shape. 
    All airfoils are then collected in a nested list."""

    airfoil1 = np.loadtxt(thickness1_file)
    airfoil2 = np.loadtxt(thickness2_file)
    airfoil3 = np.loadtxt(thickness3_file)
    airfoil4 = np.loadtxt(thickness4_file)
    airfoil5 = np.loadtxt(thickness5_file)
    airfoil6 = np.loadtxt(cylinder_file)
    airfoils = [airfoil1,airfoil2,airfoil3,airfoil4,airfoil5,airfoil6]

    return airfoils

def initialize_arrays(N, B, length):
    # Initialize all arrays using np.empty
    thetas = np.empty((N, B))
    thetas[0] = [0, 2*np.pi/B, 4*np.pi/B]

    U_turb = np.zeros((3, length))
    velocities = np.empty((B, N, 3, length))
    velocities_in4 = np.empty((B, N, 3, length))

    p = np.empty((B, N, length, 2))  # p[..., 0] for y, p[..., 1] for z
    p_y = p[..., 0]
    p_z = p[..., 1]

    r_array = np.empty((B, N, 3, length))

    W_qs_old = np.zeros((B, length, 2))  # W_qs_old[..., 0] for y, W_qs_old[..., 1] for z
    W_qs_y_old = W_qs_old[..., 0]
    W_qs_z_old = W_qs_old[..., 1]

    W_int_old = np.zeros((B, length, 2))
    W_int_y_old = W_int_old[..., 0]
    W_int_z_old = W_int_old[..., 1]

    W = np.zeros((N, B, length, 2))  # W_old[..., 0] for y, W_old[..., 1] for z
    W_y = W[..., 0]
    W_z = W[..., 1]

    fs_old = np.empty((B, length))

    f_g = np.empty(length)

    Power = np.empty(N)
    Thrust = np.empty(N)
    Thrust1 = np.empty(N)
    Thrust2 = np.empty(N)
    Thrust3 = np.empty(N)
    theta_pitch = np.empty((N, B))
    time = np.empty(N)

    # Return all arrays
    return (
        thetas, U_turb, velocities, velocities_in4,
        p_y, p_z, r_array,
        W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old,
        W_y, W_z, fs_old, f_g,
        Power, Thrust1, Thrust2, Thrust3, Thrust, theta_pitch, time
    )