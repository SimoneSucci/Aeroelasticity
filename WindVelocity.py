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
import scipy.signal as ss
from scipy.interpolate import RegularGridInterpolator


#Fixing all the path so it works from any terminal
FILE_DIR = Path(__file__).parent  # directory where this file is located 
FUNCTION_DIR = FILE_DIR / 'functions'
#sys.path.append(str(FUNCTION_DIR))
DATA_DIR = (FILE_DIR / 'data')

###### SWITCHES ########

Tower = False
Shear = False 
Dynamic_wake = False
Dynamic_stall = False
Turbulence = False
Yaw_model = False

omega = 0.69  # angular velocity
omega0 = 0.5
dt = 0.3   # time step
N = 1000   # number of iterations

B = 3   # number of blades
V_hub = 15   # wind speed at hub height

rho =1.225
H = 119   # hub height
L = 7.1   # shaft
R = 89.17  # blade radius

theta_tilt = 0   # in rad
theta_cone = 0
theta_yaw = 0 
pitch_value = 0   # should be in degrees
switch1 = 100
switch2 = 150

x_blade = 70

a_tower = 3.32   # radius used for tower shadow
nu = 0.2   # shear exponent for wind shear
k = 0.6 #dynamic wake model (Øye)

dx = 7
dy = dx
dz = V_hub*dt


Cp_opt = 0.467
lam_opt = 7.97
pitch_opt = np.deg2rad(-0.076) #rad
P_rated = 10.64*10**6 #W
A = np.pi*R**2 #m^2
omega_rated = ((2*lam_opt**3*P_rated)/(R**3*A*rho*Cp_opt))**(1/3) #rad/s
V0_rated = omega_rated*R/lam_opt #m/s
omega_ref = omega_rated*1.02 #rad/s

Inertia_rotor = 1.6*10**8 #kgm^2
KI = 0.64 #rad/rad
KP = 1.5 #rad/(rad/s)
KK = 14 #deg
theta_min = 0 #deg
theta_max = 90 #deg
K = 0.5*rho*R**3/lam_opt**3*A*Cp_opt

def update_pitch(theta_pitch, thetaI_old, omeg, omega_ref, KK, KP, KI, dt, theta_min, theta_max):
    GK = 1/(1+theta_pitch/KK)
    thetaP = GK*KP*(omeg-omega_ref)
    thetaI = thetaI_old + GK*KI*(omeg-omega_ref)*dt
    thetaSP = thetaP + thetaI

    thetaI = max(thetaI, theta_min)
    thetaI = min(thetaI, theta_max)
    thetaSP = max(thetaSP, theta_min)
    thetaSP = min(thetaSP, theta_max)

    return thetaI, thetaSP

def update_omega(omega, dt, P_rated, K, omega_rated, M_aero, I_rotor):
    if omega<omega_rated:
        MG = K*omega**2
    else:
        MG = P_rated/omega

    omega_next = omega + (M_aero-MG)/I_rotor*dt

    return MG, omega_next



import functions.initialize as Init
import functions.Positions as Positions
import functions.Winds as Winds
import functions.Plotting as Plots
import functions.ashes as ashes

radii, chords, betas, thicknesses, length = Init.load_blade_data(DATA_DIR /"bladedat.txt")

airfoils = Init.load_airfoils(
    DATA_DIR / 'FFA-W3-241_ds.txt',
    DATA_DIR / 'FFA-W3-301_ds.txt',
    DATA_DIR / 'FFA-W3-360_ds.txt',
    DATA_DIR / 'FFA-W3-480_ds.txt',
    DATA_DIR / 'FFA-W3-600_ds.txt',
    DATA_DIR / 'cylinder_ds.txt'
)

mann_box = Winds.build_turbulence_box((32, 32, N), (dx, dy, dz), V_hub)

def simulate_wind_velocity(theta_cone: float,
                  theta_yaw: float,
                  theta_tilt: float,
                  omega0: float,
                  dt: float,
                  N: int,
                  V_hub: float,
                  )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loop in time to find the angular positions of the blades, their velocities, 
    and the loads due to induced wind."""
    thetas, U_turb, velocities, velocities_in4, p_y, p_z, r_array, W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old, W_y, W_z, fs_old, f_g, Torque, Power, Thrust1, Thrust2, Thrust3, Thrust, theta_pitch, time, thetas_pitch, omegas, Power_G = Init.initialize_arrays(N, B, length)
    theta_pitch_new = 0
    thetaI_old = 0
    omega_new = omega0
    for i in range(0,N):
        time[i] = i*dt
        thetas_pitch[i] = theta_pitch_new
        omegas[i] = omega_new
        if i<N-1:
            thetas[i+1] = np.array([thetas[i,0]+omegas[i]*dt, thetas[i,1]+omegas[i]*dt, thetas[i,2]+omegas[i]*dt])

        for j in range(B):
            theta = thetas[i,j]
            a23 = Positions.build_matrix_a23(theta) #update matrix for each blade
            a14= Positions.build_matrix_a14(theta_cone, theta_tilt, theta_yaw, a23)
           
            
            r_array[j,i] = Positions.get_position(radii,Positions.build_matrices_notime(theta_cone, theta_tilt, theta_yaw)[0], a14, H, L)

            if Turbulence:
                U_turb[i,j] = Winds.interpolate_turbulence_box(mann_box,r_array[j,i],length, H, V_hub, time[i])
            velocities[j,i] = Winds.get_constant_wind(r_array[j,i,0], V_hub, length) + U_turb[i,j]
            
            if Shear: 
                velocities[j,i] = Winds.get_wind_shear(r_array[j,i,0], V_hub, H, nu) + U_turb[i,j]

            if Tower:
                velocities[j,i] = Winds.get_tower_speed(velocities[j,i], r_array[j,i], a_tower, H)
                
            velocities_in4[j,i] = np.dot(a14,velocities[j,i])

            V0_y = velocities_in4[j,i,1]
            V0_z = velocities_in4[j,i,2]

            V_rel_y = V0_y + W_y[i-1, j] - omegas[i]*radii*np.cos(theta_cone)
            V_rel_z = V0_z + W_z[i-1, j]
            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan((V_rel_z/(-V_rel_y)))
           # theta_pitch[i] = Init.get_pitch(time[i], switch1, switch2, pitch_value)
            pitch = np.ones(length)*thetas_pitch[i]
            alpha= np.rad2deg(phi)-(betas+pitch)
            
            
            
            coeff = Init.interpolate(alpha, cl_interp , cd_interp, cl_inv_interp , cl_fs_interp , fs_interp, thicknesses, length, Dynamic_stall) 
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
            
            a = (-W_z[i-1,j]/V_hub)
            
            for idx,a_loop in enumerate(a):
                if a_loop<=1/3:
                    f_g[idx] = 1
                else:
                    f_g[idx] = (1/4)*(5-3*a_loop)
            F = (2/np.pi)*(np.arccos(np.exp((-B*(np.ones(len(radii))*R-radii))/(2*radii*np.sin(np.abs(phi)))))) 
        

            Norm = np.sqrt(V0_y**2+(V0_z+f_g*W_z[i-1, j])**2)
            W_qs_z = (-B*l*np.cos(phi)/(4*np.pi*rho*radii*F*Norm))
            W_qs_y = (-B*l*np.sin(phi)/(4*np.pi*rho*radii*F*Norm))

            if theta_yaw != 0 and Yaw_model:
                n = np.array([0,np.sin(theta_yaw), np.cos(theta_yaw)])
                Vp = velocities_in4[j,i] + W_z[i-1, j]
                xi=np.arccos(np.dot(n, Vp)/np.linalg.norm(Vp, axis=0))
                W_qs_y = W_qs_y*(1+radii/R*np.tan(xi/2)*np.cos(theta-theta0))
                W_qs_z = W_qs_z*(1+radii/R*np.tan(xi/2)*np.cos(theta-theta0))

            if Dynamic_wake:
                tau1 = 1.1/(1-1.3*a)*(np.ones(len(radii))*R)/V_hub
                tau2 = (0.39-0.26*(radii/(np.ones(len(radii))*R))**2)*tau1

                H_y = W_qs_y + k*tau1*((W_qs_y-W_qs_y_old[j])/dt)
                W_int_y = H_y+(W_int_y_old[j]-H_y)*np.exp(-dt/tau1)
                W_y[i,j] = W_int_y+(W_y[i-1, j]-W_int_y)*np.exp(-dt/tau2)

                H_z = W_qs_z + k*tau1*((W_qs_z-W_qs_z_old[j])/dt)
                W_int_z = H_z+(W_int_z_old[j]-H_z)*np.exp(-dt/tau1)
                W_z[i,j] = W_int_z+(W_z[i-1, j]-W_int_z)*np.exp(-dt/tau2)

                W_int_y_old[j] = W_int_y
                W_int_z_old[j] = W_int_z
                W_qs_y_old[j] = W_qs_y
                W_qs_z_old[j] = W_qs_z
            else:
                W_y[i, j] = W_qs_y
                W_z[i, j] = W_qs_z
           
        p_y[:,:,-1] = 0
        p_z[:,:,-1] = 0
        if Turbulence:
            p_y[:,:,-2] = 0
            p_z[:,:,-2] = 0

        Torque[i] = (np.trapz(p_y[0, i, :]*radii, radii) + np.trapz(p_y[1, i, :]*radii, radii) + np.trapz(p_y[2, i, :]*radii, radii))
        Power[i] = omegas[i]* Torque[i]
        Thrust1[i] =  np.trapz(p_z[0,i,:], radii)
        Thrust2[i] = np.trapz(p_z[1,i,:], radii)
        Thrust3[i] = np.trapz(p_z[2,i,:], radii)
        Thrust[i] = Thrust1[i] + Thrust2[i]  + Thrust3[i]
        MG, omega_new= update_omega(omegas[i], dt, P_rated, K, omega_rated, Torque[i], Inertia_rotor)

        Power_G[i] = omegas[i]*MG
 
        thetaI_old, theta_pitch_new = update_pitch(thetas_pitch[i], thetaI_old, omegas[i], omega_ref, KK, KP, KI, dt, theta_min, theta_max)


        
    return time, thetas, r_array, velocities_in4, p_y, p_z, Power, Thrust1, Thrust2, Thrust3, Thrust, W_y, W_z, omegas, thetas_pitch, Power_G


#Create plots
# theta_yaw = np.deg2rad(20)
# Dynamic_wake = True
# Yaw_model = True
# theta0 = Init.define_theta0(theta_tilt, theta_yaw)

# cl_interp , cd_interp, cl_inv_interp , cl_fs_interp , fs_interp = Init.pre_interpolate(airfoils)
Turbulence = True
time, angles, positions, speeds, pys, pzs, P, T1, T2, T3, T, Wy, Wz, omega_array, pitch_array, PG_array = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt, omega0, dt, N, V_hub)
plt.plot(time, omega_array)
plt.show()
plt.plot(time, pitch_array)
plt.show()
plt.plot(time[300:], PG_array[300:])
plt.plot(time[300:], P[300:], label='Mech')
plt.legend()
plt.show()




