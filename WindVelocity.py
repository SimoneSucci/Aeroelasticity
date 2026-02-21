#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:55:12 2026

@author: ombeline
"""

import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


Tower = True
Shear = True 
Dynamic_wake = False

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

radii, chords, betas, thicknesses, length = load_blade_data("bladedat.txt")


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

airfoils = load_airfoils('FFA-W3-241_ds.txt', 'FFA-W3-301_ds.txt', 'FFA-W3-360_ds.txt', 'FFA-W3-480_ds.txt', 'FFA-W3-600_ds.txt', 'cylinder_ds.txt')

B = 3   # number of blades
V_hub = 8   # wind speed at hub height

rho =1.225
H = 119   # hub height
L = 7.1   # shaft
R = 89.17  # blade radius

theta_tilt = 0   # in rad
theta_cone = 0
theta_yaw = 0
theta_pitch = [0,0,0]   # should be in degrees

x_blade = 70

omega = 0.72   # angular velocity
dt = 0.15   # time step
N = 500   # number of iterations

a_tower = 3.32   # radius used for tower shadow
nu = 0.2   # shear exponent for wind shear
k = 0.6 #dynamic wake model (Øye)


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
    a23 = np.array([[np.cos(theta_blade), np.sin(theta_blade),0], 
                     [-np.sin(theta_blade), np.cos(theta_blade), 0],
                     [0,0,1]])
    return a23

def build_matrix_a14(theta_cone: float, 
                     theta_tilt: float,
                     theta_yaw: float, 
                     a23: np.ndarray
                     ) -> np.ndarray:
    """Builds transformation matrix from frame 1 to 4, depends on time through a23"""
    a12, a34 = build_matrices_notime(theta_cone, theta_tilt, theta_yaw)
    a14 = np.dot(a34,np.dot(a23,a12))
    return a14

def get_position(r: Union[float, np.ndarray], 
                 a12: np.ndarray,
                 a14: np.ndarray
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

def get_constant_wind(x: Union[float, np.ndarray],
                      V_hub: float
                  ) -> np.ndarray:
    """Outputs the velocity vector for a constant wind velocity in the z direction."""
    V0_array = np.ones(len(x))*V_hub
    return np.array([np.zeros(length),np.zeros(length),V0_array])

def get_wind_shear(x: Union[float, np.ndarray], 
               V_hub: float
               ) -> np.ndarray:
    """Outputs the velocity vector in the case of wind shear. 
    x should be the vertical position in frame 1."""

    v_shear = np.array([np.zeros_like(x),np.zeros_like(x),V_hub*(x/H)**nu])
    return v_shear
    
def get_tower_speed(V0: Union[float, np.ndarray], 
                coord: np.ndarray
                ) -> np.ndarray:
    """Outputs velocity array at each point in coord: for input wind V0, and with tower shadow effect."""

    V0  = V0[2]
    x, y, z = coord[0], coord[1], coord[2]
    if np.all(x)<H:
        a = a_tower
    else:
        a=0
    r = np.sqrt(y**2+z**2)
    Vr = z/r*V0*(1-(a/r)**2)
    Vt = y/r*V0*(1+(a/r)**2)
    
    Vel = np.array([np.zeros_like(x),y/r*Vr-z/r*Vt, z/r*Vr+y/r*Vt ])

    return Vel

def pre_interpolate(airfoils: List
                    ) -> Tuple[List, List, List, List]:
    """interpolate the cl and cd values to the different thicknesses"""
    cl_inv_thick = [] #initialise
    cl_fs_thick = []
    fs_thick = []
    cdthick = []

    for foil in airfoils: # k indicates the airfoil
        cl_inv_thick.append(interp1d(foil[:,0],foil[:,5], kind="linear", bounds_error=False, fill_value="extrapolate"))
        cl_fs_thick.append(interp1d(foil[:,0],foil[:,6], kind="linear", bounds_error=False, fill_value="extrapolate"))
        fs_thick.append(interp1d(foil[:,0],foil[:,4], kind="linear", bounds_error=False, fill_value="extrapolate"))

        cdthick.append(interp1d(foil[:,0], foil[:,2], kind="linear", bounds_error=False, fill_value="extrapolate"))

    return cdthick, fs_thick, cl_inv_thick, cl_fs_thick

def interpolate(alpha: Union[float, np.ndarray], 
                cl_inv_thick: List, 
                cl_fs_thick: List,
                fs_thick: List,
                cdthick: List, 
                thicknesses: np.ndarray
                ) -> dict:
    """interpolate the lift and drag coefficients to the angles of attack"""

    cl_inv = np.zeros(length)
    cl_fs = np.zeros(length)
    fs_stat = np.zeros(length)
    cd = np.zeros(length)
    for idx, a in enumerate(alpha):
        cl_inv_temps = np.array([f(a) for f in cl_inv_thick])   # shape (6,)
        cl_fs_temps = np.array([f(a) for f in cl_fs_thick])   # shape (6,)
        fs_temps = np.array([f(a) for f in fs_thick])   # shape (6,)
        cd_temps = np.array([f(a) for f in cdthick]) 
        
        #then interpolate to the actual thickness
        thick_prof=np.array([100,60,48,36,30.1,24.1])
        order = np.argsort(thick_prof)           # ascending order
        thick_sorted = thick_prof[order]
        clift_inv=interp1d(thick_sorted[:],cl_inv_temps[:])
        clift_fs=interp1d(thick_sorted[:],cl_fs_temps[:])
        fs_interp=interp1d(thick_sorted[:],fs_temps[:])

        cdrag=interp1d(thick_sorted[:],cd_temps[:])
        cl_inv[idx] = clift_inv(thicknesses[idx])
        cl_fs[idx] = clift_fs(thicknesses[idx])
        fs_stat[idx] = fs_interp(thicknesses[idx])
        cd[idx] = cdrag(thicknesses[idx])
    return {"Cl_inv": cl_inv,"Cl_fs": cl_fs,"fs_stat": fs_stat, "Cd": cd}


def simulate_wind_velocity(theta_cone: float,
                  theta_yaw: float,
                  theta_tilt: float,
                  omega: float,
                  dt: float,
                  N: int,
                  V_hub: float,
                  )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loop in time to find the angular positions of the blades, their velocities, 
    and the loads due to induced wind."""

    thetas = np.zeros((N,B))
    thetas[0] = 0,2*np.pi/B, 4*np.pi/3
    
    shear_vel = np.zeros((B,N,3, length))
    velocities = np.zeros((B,N,3, length))
    velocities_in4 = np.zeros((B,N,3, length))
    p_y = np.zeros((B,N,length))
    p_z = np.zeros((B,N,length))

    r_array = np.zeros((B,N,3,length))

    W_qs_y_old = np.zeros((B,length))
    W_qs_z_old = np.zeros((B,length))
    W_int_y_old = np.zeros((B,length))
    W_int_z_old = np.zeros((B,length))
    W_y_old = np.zeros((B,length))
    W_z_old = np.zeros((B,length))
    fs_old = np.zeros((B,length))

    f_g = np.zeros(length)
    
    for i in range(0,N):
        time = i*dt
        if i<N-1:
            thetas[i+1] = np.array(np.linspace(thetas[i,0]+omega*dt, omega*dt+(B-1)/B*2*np.pi, B))

        for j in range(B):
            theta = thetas[i,j]
            a23 = build_matrix_a23(theta) #update matrix for each blade
            a14= build_matrix_a14(theta_cone, theta_tilt, theta_yaw, a23)
           
            
            r_array[j,i] = get_position(radii,build_matrices_notime(theta_cone, theta_tilt, theta_yaw)[0], a14)
            
            shear_vel[j,i] = get_wind_shear(r_array[j,i,0], V_hub)

            velocities[j,i] = get_constant_wind(r_array[j,i,0], V_hub)

            if Shear: 
                velocities[j,i] = get_wind_shear(r_array[j,i,0], V_hub)

            if Tower:
                velocities[j,i] = get_tower_speed(velocities[j,i], r_array[j,i])
                
            velocities_in4[j,i] = np.dot(a14,velocities[j,i])

            V0_y = velocities_in4[j,i,1]
            V0_z = velocities_in4[j,i,2]

            V_rel_y = V0_y + W_y_old[j] - omega*radii*np.cos(theta_cone)
            V_rel_z = V0_z + W_z_old[j]
            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan((V_rel_z/(-V_rel_y)))
            pitch = np.ones(length)*theta_pitch[j]
            alpha = np.rad2deg(phi)-(betas+pitch)
            
            
            coeff = interpolate(alpha, cl_inv_thick, cl_fs_thick, fs_stat_thick, cdthick, thicknesses) 
            Cl_inv, Cl_fs, fs_stat, Cd = coeff['Cl_inv'],coeff["Cl_fs"], coeff["fs_stat"], coeff["Cd"]

            tau = 4*chords/V_rel
            fs = fs_stat+(fs_old[j]-fs_stat)*np.exp(-dt/tau)
            Cl = fs*Cl_inv+(1-fs)*Cl_fs
            fs_old[j] = fs


            l = 0.5*rho*V_rel**2*chords*Cl
            d = 0.5*rho*V_rel**2*chords*Cd
            p_z[j,i] = l*np.cos(phi)+d*np.sin(phi)
            p_y[j,i] = l*np.sin(phi)-d*np.cos(phi)
            
            a = (-W_qs_z_old[j]/V_hub)
            
            for idx,a_loop in enumerate(a):
                if a_loop<=1/3:
                    f_g[idx] = 1
                else:
                    f_g[idx] = (1/4)*(5-3*a_loop)
            F = (2/np.pi)*(np.arccos(np.exp((-B*(np.ones(length)*R-radii))/(2*radii*np.sin(np.abs(phi))))))            

            Norm = np.sqrt(V0_y**2+(V0_z+f_g*W_z_old[j])**2)
            #print(l)
            W_qs_z = (-B*l*np.cos(phi)/(4*np.pi*rho*radii*F*Norm))
            W_qs_y = (-B*l*np.sin(phi)/(4*np.pi*rho*radii*F*Norm))

            W_qs_y_old[j] = W_qs_y
            W_qs_z_old[j] = W_qs_z

            if Dynamic_wake:
                tau1 = 1.1/(1-1.3*a)*(np.ones(length)*R)/V_hub
                tau2 = (0.39-0.26*(radii/np.ones(length)*R)**2)*tau1
                H_y = W_qs_y[j] + k*tau1*((W_qs_y[j]-W_qs_y_old[j])/dt)
                W_int_y = H_y+(W_int_y_old[j]-H_y)*np.exp(-dt/tau1)
                W_y = W_int_y+(W_y_old[j]-W_int_y)*np.exp(-dt/tau2)
                H_z = W_qs_z + k*tau1*((W_qs_z-W_qs_z_old[j])/dt)
                W_int_z = H_z+(W_int_z_old[j]-H_z)*np.exp(-dt/tau1)
                W_z = W_int_z+(W_z_old[j]-W_int_z)*np.exp(-dt/tau2)

                W_int_y_old[j] = W_int_y
                W_int_z_old[j] = W_int_z
                W_y_old[j] = W_y
                W_z_old[j] = W_z
            else:
                W_y_old[j] = W_qs_y_old[j]
                W_z_old[j] = W_qs_z_old[j]
            
            

    p_y[:,:,-1] = 0
    p_z[:,:,-1] = 0
        
    return thetas, r_array, velocities_in4, p_y, p_z

labels = ['No yaw', 'Yaw = 20°']
colors = ['tab:red', 'tab:cyan']
Vy = np.zeros((2,N, length))
Vz = np.zeros((2,N, length))
cdthick, cl_inv_thick, cl_fs_thick, fs_stat_thick, = pre_interpolate(airfoils) 

theta_yaw =np.deg2rad(0)
Shear = False
angles, positions, speeds, pys, pzs = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
Vy_result = speeds[0,:,1]
Vz_result = speeds[0,:,2]
blade1 = angles[:,0]
plt.plot(blade1, Vy_result, label = '$V_y$', color='tab:red')
plt.plot(blade1, Vz_result, label = '$V_z$', color = 'tab:cyan')
#plt.legend()
plt.title('Tower and no shear, yaw = 20°')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()


plt.plot(radii,pys[0,-2,:],label='py')
plt.plot(radii,pzs[0,-2,:],label='pz')
plt.legend()
plt.show()

for idx, theta_yaw in enumerate([0,np.deg2rad(20)]):
    #angles, positions, speeds, pys, pzs = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
    x_array = positions[0,:,0]
    y_array = positions[0,:,1]
    Vy[idx] = speeds[0,:,1]
    Vz[idx] = speeds[0,:,2]
    plt.plot(y_array, x_array, label=labels[idx], color=colors[idx])
        
plt.xlabel('y [m]')
plt.ylabel('x [m]')
plt.title('Trajectory of a point on Blade 1, for r = 70m')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
#plt.legend()
plt.grid()
plt.show()

blade1 = angles[:,0]
plt.plot(blade1, Vy[0], label = '$V_y$', color = 'tab:red')
plt.plot(blade1, Vz[0], label = '$V_z$', color = 'tab:cyan')
#plt.legend('$V_y$', '$V_z$')
plt.title('Tower and shear, No yaw')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()

plt.plot(blade1, Vy[1], label = '$V_y$', color = 'tab:red')
plt.plot(blade1, Vz[1], label = '$V_z$', color = 'tab:cyan')
#plt.legend()
plt.title('Tower and shear, Yaw = 20°')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()

