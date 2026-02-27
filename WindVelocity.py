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
from hipersim import MannTurbulenceField
from pathlib import Path
import sys
import os

#Fixing all the path so it works from any terminal
FILE_DIR = Path(__file__).parent  # directory where this file is located 
FUNCTION_DIR = FILE_DIR / 'functions'
sys.path.append(str(FUNCTION_DIR))
DATA_DIR = (FILE_DIR / 'data')

###### SWITCHES ########

Tower = True
Shear = True 
Dynamic_wake = False
Dynamic_stall = True
Turbulence = True

omega = 0.72   # angular velocity
dt = 0.3   # time step
N = 1000   # number of iterations

B = 3   # number of blades
V_hub = 8   # wind speed at hub height

rho =1.225
H = 119   # hub height
L = 7.1   # shaft
R = 89.17  # blade radius

theta_tilt = 0   # in rad
theta_cone = 0
theta_yaw = 0
pitch_value = 9   # should be in degrees
switch1 = 100
switch2 = 150

x_blade = 70

a_tower = 3.32   # radius used for tower shadow
nu = 0.2   # shear exponent for wind shear
k = 0.6 #dynamic wake model (Øye)

dx = 7
dy = dx
dz = V_hub*dt

from func import *
from Positions import *
from Winds import * 

radii, chords, betas, thicknesses, length = load_blade_data(DATA_DIR /"bladedat.txt")

airfoils = load_airfoils(
    DATA_DIR / 'FFA-W3-241_ds.txt',
    DATA_DIR / 'FFA-W3-301_ds.txt',
    DATA_DIR / 'FFA-W3-360_ds.txt',
    DATA_DIR / 'FFA-W3-480_ds.txt',
    DATA_DIR / 'FFA-W3-600_ds.txt',
    DATA_DIR / 'cylinder_ds.txt'
)



def pre_interpolate(airfoils: List
                    ) -> Tuple[List, List, List, List, List]:
    """interpolate the cl and cd values to the different thicknesses, all values whether dyanmic stall is on or not"""
    cl_inv_thick = [] #initialise
    cl_fs_thick = []
    fs_thick = []
    cdthick = []
    clthick = []
    for foil in airfoils: # k indicates the airfoil
        clthick.append(interp1d(foil[:,0],foil[:,1], kind="linear", bounds_error=False, fill_value="extrapolate"))
        cdthick.append(interp1d(foil[:,0], foil[:,2], kind="linear", bounds_error=False, fill_value="extrapolate"))
        cl_inv_thick.append(interp1d(foil[:,0],foil[:,5], kind="linear", bounds_error=False, fill_value="extrapolate"))
        cl_fs_thick.append(interp1d(foil[:,0],foil[:,6], kind="linear", bounds_error=False, fill_value="extrapolate"))
        fs_thick.append(interp1d(foil[:,0],foil[:,4], kind="linear", bounds_error=False, fill_value="extrapolate"))
        
    return clthick, cdthick, fs_thick, cl_inv_thick, cl_fs_thick

def interpolate(alpha: Union[float, np.ndarray], 
                clthick: List,
                cdthick: List,
                fs_thick: List, 
                cl_inv_thick: List, 
                cl_fs_thick: List,
                thicknesses: np.ndarray
                ) -> dict:
    """interpolate the lift and drag coefficients to the angles of attack, output varies depending on whether dynamic stall is on or not."""

    cl_inv = np.zeros(length)
    cl_fs = np.zeros(length)
    fs_stat = np.zeros(length)
    cd_stat = np.zeros(length)
    cl_stat = np.zeros(length)
    if Dynamic_stall:
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
            cd_stat[idx] = cdrag(thicknesses[idx])
    else: 
        for idx, a in enumerate(alpha):
            cl_temps = np.array([f(a) for f in clthick])   # shape (6,)
            cd_temps = np.array([f(a) for f in cdthick]) 
            
            #then interpolate to the actual thickness
            thick_prof=np.array([100,60,48,36,30.1,24.1])
            order = np.argsort(thick_prof)           # ascending order
            thick_sorted = thick_prof[order]

            clift=interp1d(thick_sorted[:],cl_temps[:])
            cdrag=interp1d(thick_sorted[:],cd_temps[:])

            cl_stat[idx] = clift(thicknesses[idx])
            cd_stat[idx] = cdrag(thicknesses[idx])
    return {"Cl": cl_stat, "Cd": cd_stat, "fs_stat": fs_stat, "Cl_inv": cl_inv, "Cl_fs": cl_fs}
   

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

    thetas, U_turb, velocities, velocities_in4, p_y, p_z, r_array, W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old, W_y_old, W_z_old, fs_old, f_g, Power, Thrust, theta_pitch, time = initialize_arrays(N, B, length)
    
    for i in range(0,N):
        time[i] = i*dt
        if i<N-1:
            thetas[i+1] = np.array(np.linspace(thetas[i,0]+omega*dt, omega*dt+(B-1)/B*2*np.pi, B))

        for j in range(B):
            theta = thetas[i,j]
            a23 = build_matrix_a23(theta) #update matrix for each blade
            a14= build_matrix_a14(theta_cone, theta_tilt, theta_yaw, a23)
           
            
            r_array[j,i] = get_position(radii,build_matrices_notime(theta_cone, theta_tilt, theta_yaw)[0], a14, H, L)

            if Turbulence:
                U_turb = load_turbulence_box(FILE_DIR/"mann_box_V08.nc",r_array[j,i])
            velocities[j,i] = get_constant_wind(r_array[j,i,0], V_hub, length) + U_turb
            
            if Shear: 
                velocities[j,i] = get_wind_shear(r_array[j,i,0], V_hub, H, nu) #+ U_turb

            if Tower:
                velocities[j,i] = get_tower_speed(velocities[j,i], r_array[j,i], a_tower, H)
                
            velocities_in4[j,i] = np.dot(a14,velocities[j,i])

            V0_y = velocities_in4[j,i,1]
            V0_z = velocities_in4[j,i,2]

            V_rel_y = V0_y + W_y_old[i-1, j] - omega*radii*np.cos(theta_cone)
            V_rel_z = V0_z + W_z_old[i-1, j]
            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan((V_rel_z/(-V_rel_y)))
            theta_pitch[i] = get_pitch(time[i], switch1, switch2, pitch_value)
            pitch = np.ones(length)*theta_pitch[i,j]
            alpha = np.rad2deg(phi)-(betas+pitch)
            
            
            coeff = interpolate(alpha, clthick, cdthick, fs_stat_thick, cl_inv_thick, cl_fs_thick, thicknesses) 
            Cl_stat, Cd, fs_stat, Cl_inv, Cl_fs = coeff["Cl"], coeff["Cd"], coeff["fs_stat"], coeff["Cl_inv"], coeff["Cl_fs"]

            if Dynamic_stall:
                tau = 4*chords/V_rel
                fs = fs_stat+(fs_old[j]-fs_stat)*np.exp(-dt/tau)
                Cl = fs*Cl_inv+(1-fs)*Cl_fs
                fs_old[j] = fs
            else:
                Cl = Cl_stat

            l = 0.5*rho*V_rel**2*chords*Cl
            d = 0.5*rho*V_rel**2*chords*Cd
            p_z[j,i] = l*np.cos(phi)+d*np.sin(phi)
            p_y[j,i] = l*np.sin(phi)-d*np.cos(phi)
            
            a = (-W_z_old[i,j]/V_hub)
            
            for idx,a_loop in enumerate(a):
                if a_loop<=1/3:
                    f_g[idx] = 1
                else:
                    f_g[idx] = (1/4)*(5-3*a_loop)
            F = (2/np.pi)*(np.arccos(np.exp((-B*(np.ones(len(radii))*R-radii))/(2*radii*np.sin(np.abs(phi))))))            

            Norm = np.sqrt(V0_y**2+(V0_z+f_g*W_z_old[i-1, j])**2)
            W_qs_z = (-B*l*np.cos(phi)/(4*np.pi*rho*radii*F*Norm))
            W_qs_y = (-B*l*np.sin(phi)/(4*np.pi*rho*radii*F*Norm))

            if Dynamic_wake:
                tau1 = 1.1/(1-1.3*a)*(np.ones(len(radii))*R)/V_hub
                tau2 = (0.39-0.26*(radii/(np.ones(len(radii))*R))**2)*tau1

                H_y = W_qs_y + k*tau1*((W_qs_y-W_qs_y_old[j])/dt)
                W_int_y = H_y+(W_int_y_old[j]-H_y)*np.exp(-dt/tau1)
                W_y = W_int_y+(W_y_old[i-1, j]-W_int_y)*np.exp(-dt/tau2)

                H_z = W_qs_z + k*tau1*((W_qs_z-W_qs_z_old[j])/dt)
                W_int_z = H_z+(W_int_z_old[j]-H_z)*np.exp(-dt/tau1)
                W_z = W_int_z+(W_z_old[i-1, j]-W_int_z)*np.exp(-dt/tau2)

                W_int_y_old[j] = W_int_y
                W_int_z_old[j] = W_int_z
                W_y_old[i, j] = W_y
                W_z_old[i, j] = W_z
                W_qs_y_old[j] = W_qs_y
                W_qs_z_old[j] = W_qs_z
            else:
                W_y_old[i, j] = W_qs_y
                W_z_old[i, j] = W_qs_z
           
        p_y[:,:,-1] = 0
        p_z[:,:,-1] = 0

        Power[i] = omega*(np.trapz(p_y[0, i, :]*radii, radii) + np.trapz(p_y[1, i, :]*radii, radii) + np.trapz(p_y[2, i, :]*radii, radii))
        Thrust[i] = np.trapz(p_z[0,i,:], radii) + np.trapz(p_z[1,i,:], radii) + np.trapz(p_z[2,i,:], radii)


        
    return time, thetas, r_array, velocities_in4, p_y, p_z, Power, Thrust, W_y_old, W_z_old

labels = ['No yaw', 'Yaw = 20°']
colors = ['tab:red', 'tab:cyan']
Vy = np.zeros((2,N, length))
Vz = np.zeros((2,N, length))
clthick, cdthick, fs_stat_thick, cl_inv_thick, cl_fs_thick = pre_interpolate(airfoils) 

theta_yaw =np.deg2rad(0)
Dynamic_wake = False
time, angles, positions, speeds, pys, pzs, P, T, Wy, Wz = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
Vy_result = speeds[0,:,1]
Vz_result = speeds[0,:,2]
blade1 = angles[:,0]

plt.plot(radii,pys[0,-2,:],label='py')
plt.plot(radii,pzs[0,-2,:],label='pz')
plt.legend()
plt.show()

#plt.plot(time, lift[:,0,14], label='lift')
plt.plot(time, Wz[:,0, 14], label='Wz')



Dynamic_wake = True
time, angles, positions, speeds, pys, pzs, P, T, Wy, Wz = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
Vy_result = speeds[0,:,1]
Vz_result = speeds[0,:,2]
blade1 = angles[:,0]

#plt.plot(time, lift[:,0,14], label='lift, stall')
plt.plot(time, Wz[:,0, 14], label='Wz, wake')
plt.legend()
plt.show()


plt.plot(radii,pys[0,-2,:],label='py')
plt.plot(radii,pzs[0,-2,:],label='pz')
plt.legend()
plt.show()

plt.plot(blade1, Vy_result, label = '$V_y$', color='tab:red')
plt.plot(blade1, Vz_result, label = '$V_z$', color = 'tab:cyan')
#plt.legend()
plt.title('Tower and no shear, yaw = 20°')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
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

