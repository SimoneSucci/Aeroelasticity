#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:55:12 30026

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

import functions.initialize as Init
import functions.Positions as Positions
import functions.Winds as Winds
import functions.Plotting as Plots
import functions.ashes as ashes
import functions.control as control
import functions.EomRungeKutta as RungeKutta

###### SWITCHES ########

Tower = False
Shear = False 
Dynamic_wake = True
Dynamic_stall = True
Turbulence = True
Yaw_model = False
Control = True
Gravity = True

##### VALUES ##########

omega_new = 0.5
dt = 0.3   # time step
N = 400   # number of iterations
i_cutin = 40 # time where the dynamic wake turns on (index, not sec)


B = 3   # number of blades


V_hub = 15  # wind speed at hub height


rho =1.225
H = 119   # hub height
L = 7.1   # shaft
R = 89.17  # blade radius
A = np.pi*R**2
M = 446000 #kg, mass of nacelle

theta_tilt = 0   # in rad
theta_cone = 0
theta_yaw = 0 
pitch_value = 0   # should be in degrees
switch1 = 300
switch2 = 150

x_blade = 70

a_tower = 3.32   # radius used for tower shadow
nu = 0.2   # shear exponent for wind shear
k = 0.6 #dynamic wake model (Øye)


#dx = 7
#dy = dx
#dz = V_hub*dt


Cp_opt = 0.467
lam_opt = 7.97
pitch_opt = np.deg2rad(-0.076) #rad
P_rated = 10.64*10**6 #W
A = np.pi*R**2 #m^2
omega_rated = ((2*lam_opt**3*P_rated)/(R**3*A*rho*Cp_opt))**(1/3) #rad/s
V0_rated = omega_rated*R/lam_opt #m/s
omega_ref = omega_rated*1.01 #rad/s


Inertia_rotor = 1.6*10**8 #kgm^2
KI = 0.64 #rad/rad
KP = 1.5 #rad/(rad/s)
KK = 14 #deg
theta_min = 0 #deg
theta_max = 90 #deg
K = 0.5*rho*R**3*A*Cp_opt/(lam_opt**3)
print(K-0.300131*1e8)


radii, chords, betas, thicknesses, length = Init.load_blade_data(DATA_DIR /"bladedat.txt")
#mann_box = Winds.build_turbulence_box((32, 32, N), (dx, dy, dz), V_hub)


airfoils = Init.load_airfoils(
    DATA_DIR / 'FFA-W3-241_ds.txt',
    DATA_DIR / 'FFA-W3-301_ds.txt',
    DATA_DIR / 'FFA-W3-360_ds.txt',
    DATA_DIR / 'FFA-W3-480_ds.txt',
    DATA_DIR / 'FFA-W3-600_ds.txt',
    DATA_DIR / 'cylinder_ds.txt'
)

def simulate_wind_velocity(theta_cone: float,
                  theta_yaw: float,
                  theta_tilt: float,
                  omega_new: float,
                  dt: float,
                  N: int,
                  V_hub: float,
                  )-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loop in time to find the angular positions of the blades, their velocities, 
    and the loads due to induced wind."""
    thetas, U_turb, velocities, velocities_in4, p_y, p_z, r_array, W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old, W_y, W_z, fs_old, f_g, Torque, Power, Thrust1, Thrust2, Thrust3, Thrust, time, thetas_pitch, omegas, Power_G, theta_pitch_new, thetaI_old, Cp = Init.initialize_arrays(N, B, length)
   
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

            if Vibrations:
                 y_arr[i+1], y_d_arr[i+1], y_dd_arr[i+1]= RungeKutta.rungeKutta(dt, time[i], y_arr[i], y_d_arr[i+1], y_dd_arr[i+1], M, m, )
                 u_blade = q1dot*np.array([u_y_1f, u_x_1f])+q2dot*np.array([u_y_1e, u_x_1e])+q3dot*np.array([u_y_2f, u_x_2f])


            V_rel_y = V0_y + W_y[i-1, j] - omegas[i]*radii*np.cos(theta_cone)-u_blade[0]
            V_rel_z = V0_z + W_z[i-1, j] - u_blade[1]-xdot_tower
            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan((V_rel_z/(-V_rel_y)))
            #theta_pitch[i] = Init.get_pitch(time[i], switch1, switch2, pitch_value)
            
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

            if Gravity:
                p_grav = np.array([0,0,g])*m
                p_grav = np.dot(a14, p_grav)
                p_z[j,i] += p_grav[2]
                p_y[j,i] += p_grav[1]
            
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
                theta0 = Init.define_theta0(theta_tilt, theta_yaw)
                W_qs_y = W_qs_y*(1+radii/R*np.tan(xi/2)*np.cos(theta-theta0))
                W_qs_z = W_qs_z*(1+radii/R*np.tan(xi/2)*np.cos(theta-theta0))

            if Dynamic_wake:
                if i>i_cutin:
                    tau1 = 1.1/(1-1.3*a)*(np.ones(len(radii))*R)/V_hub
                    tau2 = (0.39-0.26*(radii/(np.ones(len(radii))*R))**2)*tau1
                else:
                    tau1= 0.001
                    tau2 = 0.001


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
        
        MG = control.calculate_MG(omegas[i], P_rated, K, omega_rated)

        if Control:
            omega_new= control.update_omega(omegas[i], MG, dt, Torque[i], Inertia_rotor)  
            thetaI_old, theta_pitch_new = control.update_pitch(thetas_pitch[i], thetaI_old, omegas[i], omega_ref, KK, KP, KI, dt, theta_min, theta_max)

        Power_G[i] = omegas[i]*MG
        Cp[i] = Power[i]/(0.5*rho*V_hub**3*A)

        
    return time, thetas, r_array, velocities_in4, p_y, p_z, Power, Thrust1, Thrust2, Thrust3, Thrust, W_y, W_z, omegas, thetas_pitch, Power_G, Cp, Torque

Turbulence = False
cl_interp , cd_interp, cl_inv_interp , cl_fs_interp , fs_interp = Init.pre_interpolate(airfoils)
windspeeds = np.linspace(2,20,19)
Powers_speedsweep= np.empty(len(windspeeds))
omegas_speedsweep= np.empty(len(windspeeds))
pitches_speedsweep= np.empty(len(windspeeds))
Cp_speedsweep= np.empty(len(windspeeds))


for idx, V_hub in enumerate(windspeeds):
    dx = 7
    dy = dx
    dz = V_hub*dt
    mann_box = Winds.build_turbulence_box((32, 32, N), (dx, dy, dz), V_hub)
    time, angles, positions, speeds, pys, pzs, P, T1, T2, T3, T, Wy, Wz, omega_array, pitch_array, PG_array, Cp, torque = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt, omega_new, dt, N, V_hub)
    Powers_speedsweep[idx] = P[-1]
    omegas_speedsweep[idx] = omega_array[-1]
    pitches_speedsweep[idx] = pitch_array[-1]
    Cp_speedsweep[idx] = Cp[-1]


fig, axs = plt.subplots(2,2, figsize=(9,6))

axs[0,1].plot(windspeeds, Powers_speedsweep/10**6)
axs[0,1].set_ylabel('$P_{mech}$ [MW]')
axs[0,1].set_xlabel('Wind speed [m/s]')
axs[0,1].grid()


axs[1,1].plot(windspeeds, omegas_speedsweep)
axs[1,1].set_ylabel('$\omega$ [rad/s]')
axs[1,1].set_xlabel('Wind speed [m/s]')
axs[1,1].grid()


axs[1,0].plot(windspeeds, (pitches_speedsweep))
axs[1,0].set_ylabel('Pitch Angle [deg]')
axs[1,0].set_xlabel('Wind speed [m/s]')
axs[1,0].grid()


axs[0,0].plot(windspeeds, Cp_speedsweep)
axs[0,0].set_ylabel('$C_p$')
axs[0,0].set_xlabel('Wind speed [m/s]')
axs[0,0].grid()

fig.subplots_adjust(hspace=0.5)
plt.tight_layout()
plt.show()

"""



ash_file = DATA_DIR/f'Ashes_v{V_hub}_turb.txt'
data_df, unit_dict = ashes.import_results_timesteps (ash_file)

time_ash = data_df['Time']
P_ash = data_df['Power (aero)']/1e3
Cp_ash = data_df['Power coef. (CP)']/100
RPM_ash = data_df['RPM']
pitch_ash = data_df['Representative demanded pitch angle']


fig, axs = plt.subplots(2,2, figsize=(9,6))


axs[0,0].plot(time[300:], Cp[300:], label='BEM')
axs[0,0].plot(time_ash[301:], Cp_ash[301:], label='Ashes',linestyle='--',color='r')
axs[0,0].legend()
axs[0,0].set_ylabel('$C_p$')
axs[0,0].set_xlabel('Time [s]')
axs[0,0].grid()



axs[0,1].plot(time[300:], P[300:]/10**6, label='$P_{mech}$ BEM')
axs[0,1].plot(time_ash[301:], P_ash[301:], label='$P_{mech}$ Ashes',linestyle='--',color='r')
axs[0,1].plot(time[300:], PG_array[300:]/10**6, color='y', label='$P_{elec}$ BEM')
axs[0,1].set_ylabel('Power [MW]')
axs[0,1].set_xlabel('Time [s]')
axs[0,1].legend()
axs[0,1].grid()



axs[1,0].plot(time[300:], pitch_array[300:], label="BEM")
axs[1,0].plot(time_ash[300:], pitch_ash[300:], label='Ashes',linestyle='--',color='r')
axs[1,0].legend()
axs[1,0].set_ylabel('Pitch Angle [deg]')
axs[1,0].set_xlabel('Time [s]')
axs[1,0].grid()


axs[1,1].plot(time[300:], omega_array[300:], label="BEM")
axs[1,1].plot(time_ash[300:], RPM_ash[300:]*np.pi/30, label='Ashes',linestyle='--',color='r')
axs[1,1].legend()
axs[1,1].set_ylabel('Rotational Speed [rad/s]')
axs[1,1].set_xlabel('Time [s]')
axs[1,1].grid()

plt.tight_layout()
plt.show()
"""