import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from hipersim import MannTurbulenceField
from pathlib import Path
import sys
from scipy.interpolate import RegularGridInterpolator


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

def initialize_arrays(N, B, length, DOF):
    # Initialize all arrays using np.empty
    thetas = np.empty((N, B))
    thetas[0] = [0, 2*np.pi/B, 4*np.pi/B]

    U_turb = np.zeros((N,B,3, length))
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

    fs_old = np.zeros((B, length))

    f_g = np.empty(length)

    Torque = np.empty(N)
    Power = np.empty(N)
    Cp = np.empty(N)
    Thrust = np.empty(N)
    Thrust1 = np.empty(N)
    Thrust2 = np.empty(N)
    Thrust3 = np.empty(N)
    time = np.empty(N)

    thetas_pitch = np.empty(N)
    omegas = np.empty(N)
    Power_G = np.empty(N)
    theta_pitch_new = 0
    thetaI_old = 0

    y_arr = np.zeros((N+1,DOF))
    y_d_arr = np.zeros((N+1,DOF))
    y_d_arr [0,1]=0.5
    y_dd_arr = np.zeros((N+1,DOF))
    u_blade = np.zeros((2,length))

    Mbend_y = np.zeros(N+1)
    Mbend_z = np.zeros(N+1)


    # Return all arrays
    return (
        thetas, U_turb, velocities, velocities_in4,
        p_y, p_z, r_array,
        W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old,
        W_y, W_z, fs_old, f_g,
        Torque, Power, Thrust1, Thrust2, Thrust3, Thrust, 
        time, thetas_pitch, omegas, Power_G, theta_pitch_new, thetaI_old,
        Cp, y_arr, y_d_arr, y_dd_arr, u_blade, Mbend_y, Mbend_z
    )

def pre_interpolate(airfoils: List
                    ) -> Tuple[List, List, List, List, List]:
    """interpolate the cl and cd values to the different thicknesses, all values whether dyanmic stall is on or not"""
    cl_inv_grid =[]  #initialise
    cl_fs_grid = [] 
    fs_grid = []
    cd_grid = [] 
    cl_grid= []
    for foil in (airfoils): # k indicates the airfoil
        
        cl_grid.append(foil[:,1]) 
        cd_grid.append(foil[:,2])
        cl_inv_grid.append(foil[:,5])
        cl_fs_grid.append(foil[:,6])
        fs_grid.append(foil[:,4])
        aoa = foil[:,0] 

    
    thick = np.array([24.1,30.1,36,48,60,100])

    # 3. Create the interpolator
    cl_interp = RegularGridInterpolator((thick, aoa), cl_grid)  
    cd_interp = RegularGridInterpolator((thick, aoa), cd_grid)
    cl_inv_interp = RegularGridInterpolator((thick, aoa), cl_inv_grid)
    cl_fs_interp = RegularGridInterpolator((thick, aoa), cl_fs_grid)
    fs_interp = RegularGridInterpolator((thick, aoa), fs_grid)     

    return  cl_interp , cd_interp, cl_inv_interp , cl_fs_interp , fs_interp


def interpolate(alpha: Union[float, np.ndarray], 
                cl_interp: List ,
                cd_interp: List,
                cl_inv_interp: List , 
                cl_fs_interp: List , 
                fs_interp: List,
                thicknesses: np.ndarray,
                length: int,
                Dynamic_stall: bool) -> dict:
    """interpolate the lift and drag coefficients to the angles of attack, output varies depending on whether dynamic stall is on or not."""

    cl_stat = np.zeros(length)
    cd_stat = np.zeros(length)
    cl_inv = np.zeros(length)
    cl_fs = np.zeros(length)
    fs_stat = np.zeros(length)    
    
    # Replace all NaNs with 180#
    alpha = np.nan_to_num(alpha, nan=180)
    
    # Create a 2D array of all points: [[t1, a], [t2, a], [t3, a]...]
    points = np.column_stack((thicknesses, alpha))
    
        
    if Dynamic_stall:       
        
        cl_inv = cl_inv_interp(points)
        cl_fs = cl_fs_interp(points)
        fs_stat = fs_interp(points)
        cd_stat = cd_interp(points)

    else:
    
        cl_stat = cl_interp(points)
        cd_stat = cd_interp(points)

    return {"Cl": cl_stat, "Cd": cd_stat, "fs_stat": fs_stat, "Cl_inv": cl_inv, "Cl_fs": cl_fs}

def define_theta0(theta_tilt, theta_yaw):
    if theta_tilt == 0:
        if theta_yaw > 0:
            theta0 = np.deg2rad(90)
        else:
            theta0 = np.deg2rad(270)
    else:   
        theta0= np.arctan(-np.tan(theta_yaw)/np.sin(theta_tilt))
    return theta0

