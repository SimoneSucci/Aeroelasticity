#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:55:12 30026

@author: ombeline
"""
#%%
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



##### VALUES ##########
DOF = 5

omega_new = 0.5
dt = 0.1   # time step
N = 1000   # number of iterations
i_cutin = 500 # time where the dynamic wake turns on (index, not sec)


B = 3   # number of blades


V_hub = 12  # wind speed at hub height

g = 9.81 #gravity constant m/s^2
rho =1.225 #air density kg/m^3
H = 119   # hub height
L = 7.1   # shaft
R = 89.17  # blade radius
A = np.pi*R**2
M = 446000 #kg, mass of nacelle
k_tow =  1.7*10**6 #Spriing constant for tower deflection N/m

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


radii, chords, betas, thicknesses, length = Init.load_blade_data(DATA_DIR /"bladedat.txt")


radii, u_y_1f, u_z_1f, u_y_1e, u_z_1e, u_y_2f, u_z_2f, m = np.loadtxt(DATA_DIR/"modeshapes.txt", unpack=True)

omegas_modes = [3.93,6.10,11.28]



#mann_box = Winds.build_turbulence_box((32, 32, N), (dx, dy, dz), V_hub)


airfoils = Init.load_airfoils(
    DATA_DIR / 'FFA-W3-241_ds.txt',
    DATA_DIR / 'FFA-W3-301_ds.txt',
    DATA_DIR / 'FFA-W3-360_ds.txt',
    DATA_DIR / 'FFA-W3-480_ds.txt',
    DATA_DIR / 'FFA-W3-600_ds.txt',
    DATA_DIR / 'cylinder_ds.txt'
)

#%%
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
    thetas, U_turb, velocities, velocities_in4, p_y, p_z, r_array, W_qs_y_old, W_qs_z_old, W_int_y_old, W_int_z_old, W_y, W_z, fs_old, f_g, Torque, Power, Thrust1, Thrust2, Thrust3, Thrust, time, thetas_pitch, omegas, Power_G, theta_pitch_new, thetaI_old, Cp, y_arr, y_d_arr, y_dd_arr, u_blades, Mbend_y, Mbend_z,xtower, xtowerd,tip_deflection = Init.initialize_arrays(N, B, length, DOF)

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
            V_rel_z = V0_z + W_z[i-1, j] -xtowerd[i]*np.ones(length)

            if DOF==11 or j==0:
                V_rel_y = V_rel_y - u_blades[j,0]
                V_rel_z = V_rel_z - u_blades[j,1]
                

            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan2(V_rel_z,-V_rel_y)
            
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
                row = np.array([0, g, 0])  
                p_grav = np.tile(row, (18, 1)).T*m
                p_grav = np.dot(a14, p_grav)
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
            #omega_new= control.update_omega(omegas[i], MG, dt, Torque[i], Inertia_rotor)  
            thetaI_old, theta_pitch_new = control.update_pitch(thetas_pitch[i], thetaI_old, omegas[i], omega_ref, KK, KP, KI, dt, theta_min, theta_max)

        Power_G[i] = omegas[i]*MG
        Cp[i] = Power[i]/(0.5*rho*V_hub**3*A)

        
        if Vibrations:
            pitch_angle = np.deg2rad(thetas_pitch[i])
            u_y_1f_pitched = u_y_1f*np.cos(pitch_angle)+u_z_1f*np.sin(pitch_angle)
            u_z_1f_pitched = -u_y_1f*np.sin(pitch_angle)+u_z_1f*np.cos(pitch_angle)
            u_y_1e_pitched = u_y_1e*np.cos(pitch_angle)+u_z_1e*np.sin(pitch_angle)
            u_z_1e_pitched = -u_y_1e*np.sin(pitch_angle)+u_z_1e*np.cos(pitch_angle)
            u_y_2f_pitched = u_y_2f*np.cos(pitch_angle)+u_z_2f*np.sin(pitch_angle)
            u_z_2f_pitched = -u_y_2f*np.sin(pitch_angle)+u_z_2f*np.cos(pitch_angle)
            modes = np.array([u_y_1f_pitched, u_z_1f_pitched, u_y_1e_pitched, u_z_1e_pitched, u_y_2f_pitched, u_z_2f_pitched])

            y_arr[i+1], y_d_arr[i+1], y_dd_arr[i+1]= RungeKutta.rungeKutta(DOF,dt, y_arr[i], y_d_arr[i], y_dd_arr[i], M,k_tow, m, Inertia_rotor, radii,Thrust[i], Torque[i],MG, p_y[:,i], p_z[:,i],omegas_modes,modes)
        

            if DOF ==5:
                xtower[i+1],_,q1,q2,q3 = y_arr[i+1]
                deflections = q1*np.array([u_y_1f_pitched, u_z_1f_pitched])+q2*np.array([u_y_1e_pitched, u_z_1e_pitched])+q3*np.array([u_y_2f_pitched, u_z_2f_pitched])
                tip_deflection[i+1] = deflections[:,-1]

                xtowerd[i+1],omega_new,q1d,q2d,q3d = y_d_arr[i+1]
                u_blades[0] = q1d*np.array([u_y_1f_pitched, u_z_1f_pitched])+q2d*np.array([u_y_1e_pitched, u_z_1e_pitched])+q3d*np.array([u_y_2f_pitched, u_z_2f_pitched])
           
                _,_,q1dd,q2dd,q3dd = y_dd_arr[i+1]
                udd_blade = q1dd*np.array([u_y_1f_pitched, u_z_1f_pitched])+q2dd*np.array([u_y_1e_pitched, u_z_1e_pitched])+q3dd*np.array([u_y_2f_pitched, u_z_2f_pitched])
                Mbend_z[i+1] = np.trapz((radii-radii[0])*(p_z[0,i]-m*udd_blade[1]), radii)
                Mbend_y[i+1] = np.trapz((radii-radii[0])*(p_y[0,i]-m*udd_blade[0]), radii)

            elif DOF ==11:
                xtower[i+1],_,q11,q12,q13,q21,q22,q23,q31,q32,q33 = y_arr[i+1]
                deflections1 = q11*np.array([u_y_1f_pitched, u_z_1f_pitched])+q12*np.array([u_y_1e_pitched, u_z_1e_pitched])+q13*np.array([u_y_2f_pitched, u_z_2f_pitched])
                deflections2 = q21*np.array([u_y_1f_pitched, u_z_1f_pitched])+q22*np.array([u_y_1e_pitched, u_z_1e_pitched])+q23*np.array([u_y_2f_pitched, u_z_2f_pitched])
                deflections3 = q31*np.array([u_y_1f_pitched, u_z_1f_pitched])+q32*np.array([u_y_1e_pitched, u_z_1e_pitched])+q33*np.array([u_y_2f_pitched, u_z_2f_pitched])
                tip_deflection[i+1] = np.array([deflections1[:,-1], deflections2[:,-1],deflections3[:,-1]])

                xtowerd[i+1], omega_new, q1d, q2d, q3d,q21d, q22d, q23d, q31d, q32d, q33d = y_d_arr[i+1]
                u_blade1 = q1d*np.array([u_y_1f_pitched, u_z_1f_pitched])+q2d*np.array([u_y_1e_pitched, u_z_1e_pitched])+q3d*np.array([u_y_2f_pitched, u_z_2f_pitched])
                u_blade2 = q21d*np.array([u_y_1f_pitched, u_z_1f_pitched])+q22d*np.array([u_y_1e_pitched, u_z_1e_pitched])+q23d*np.array([u_y_2f_pitched, u_z_2f_pitched])
                u_blade3 = q31d*np.array([u_y_1f_pitched, u_z_1f_pitched])+q32d*np.array([u_y_1e_pitched, u_z_1e_pitched])+q33d*np.array([u_y_2f_pitched, u_z_2f_pitched])
                u_blades = np.array([u_blade1, u_blade2, u_blade3])

                _,_,q1dd,q2dd,q3dd,q21dd,q22dd,q23dd,q31dd,q32dd,q33dd = y_dd_arr[i+1]
                udd_blade1 = q1dd*np.array([u_y_1f_pitched, u_z_1f_pitched])+q2dd*np.array([u_y_1e_pitched, u_z_1e_pitched])+q3dd*np.array([u_y_2f_pitched, u_z_2f_pitched])
                udd_blade2 = q21dd*np.array([u_y_1f_pitched, u_z_1f_pitched])+q22dd*np.array([u_y_1e_pitched, u_z_1e_pitched])+q23dd*np.array([u_y_2f_pitched, u_z_2f_pitched])
                udd_blade3 = q31dd*np.array([u_y_1f_pitched, u_z_1f_pitched])+q32dd*np.array([u_y_1e_pitched, u_z_1e_pitched])+q33dd*np.array([u_y_2f_pitched, u_z_2f_pitched])

                Mbend_z1 = np.trapz((radii-radii[0])*(p_z[0,i]-m*udd_blade1[1]), radii)
                Mbend_y1 = np.trapz((radii-radii[0])*(p_y[0,i]-m*udd_blade1[0]), radii)
                Mbend_z2 = np.trapz((radii-radii[0])*(p_z[1,i]-m*udd_blade2[1]), radii)
                Mbend_y2 = np.trapz((radii-radii[0])*(p_y[1,i]-m*udd_blade2[0]), radii)
                Mbend_z3 = np.trapz((radii-radii[0])*(p_z[2,i]-m*udd_blade3[1]), radii)
                Mbend_y3 = np.trapz((radii-radii[0])*(p_y[2,i]-m*udd_blade3[0]), radii)

                Mbend_y[i+1]= np.array([Mbend_y1, Mbend_y2, Mbend_y3])
                Mbend_z[i+1]=np.array([Mbend_z1, Mbend_z2, Mbend_z3])

            #plt.plot(radii, deflections[0,:])

    return time, thetas, r_array, velocities_in4, p_y, p_z, Power, Thrust1, Thrust2, Thrust3, Thrust, W_y, W_z, omegas, thetas_pitch, Power_G, Cp, Torque, tip_deflection, xtower, xtowerd, Mbend_y, Mbend_z


###### SWITCHES ########

Tower = False
Shear = False 
Dynamic_wake = True
Dynamic_stall = True
Turbulence = False
Yaw_model = False
Control = True
Gravity = True
Vibrations = True

DOF = 5
V_hub = 7

dx = 7
dy = dx
dz = V_hub*dt

cl_interp , cd_interp, cl_inv_interp , cl_fs_interp , fs_interp = Init.pre_interpolate(airfoils)
mann_box = Winds.build_turbulence_box((32, 32, N), (dx, dy, dz), V_hub)

time, angles, positions, speeds, pys, pzs, P, T1, T2, T3, T, Wy, Wz, omega_array, pitch_array, PG_array, Cp, torque, tip_deflection, xtower, xtowerd, Mbend_y, Mbend_z= simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt, omega_new, dt, N, V_hub)

#%%
fig,axs = Plots.plots_assignment2(dt,time, tip_deflection[:,0,0],'u_{y, Tip}','Deflections [m]', variable2= tip_deflection[:,0,1], legend2='u_{z, Tip}', t_start=500)
fig,axs = Plots. plots_assignment2(dt,time, xtower,'x_tower','Tower Delfection [m]')
fig,axs = Plots.plots_assignment2(dt,time, Mbend_y[:,0],'M_y', 'Bending Moment [Nm]', Mbend_z[:,0], 'M_z')

"""
plt.figure()
plt.plot(time, omega_array)
plt.figure()
plt.plot(time, pitch_array)
"""
# %%
