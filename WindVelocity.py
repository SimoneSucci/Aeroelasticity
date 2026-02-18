#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:55:12 2026

@author: ombeline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Tower = False
Shear = False 


blade_data = np.loadtxt("bladedat.txt")
radii = blade_data[:,0]
chords = blade_data[:,1]
betas = blade_data[:,2]
thicknesses = blade_data[:,3]
length = len(blade_data)

#open the files with the airfoil data
airfoil6 = np.loadtxt('cylinder_ds.txt')
airfoil1 = np.loadtxt('FFA-W3-241_ds.txt')
airfoil2 = np.loadtxt('FFA-W3-301_ds.txt')
airfoil3 = np.loadtxt('FFA-W3-360_ds.txt')
airfoil4 = np.loadtxt('FFA-W3-480_ds.txt')
airfoil5 = np.loadtxt('FFA-W3-600_ds.txt')
airfoils = [airfoil1,airfoil2,airfoil3,airfoil4,airfoil5,airfoil6]

B = 3

rho =1.225
H = 119 # hub height
L = 7.1 #shaft
R = 89.15 #blade radius
theta_tilt = 0#np.deg2rad(-5)
theta_cone = 0
theta_yaw = 0
theta_pitch = [0,0,0] #should be in degrees
x_blade = 70

omega = 0.72 #angular velocity
dt = 0.15 #time step
N = 500
a_tower = 3.32
V_hub = 8
nu = 0.2

def matrices_notime(theta_cone, theta_tilt, theta_yaw):
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
    
    
def matrix_a23(theta_blade):
    a23 = np.array([[np.cos(theta_blade), np.sin(theta_blade),0], 
                     [-np.sin(theta_blade), np.cos(theta_blade), 0],
                     [0,0,1]])
    return a23

def matrix_a14(theta_cone, theta_tilt,theta_yaw, a23):
    a12, a34 = matrices_notime(theta_cone, theta_tilt, theta_yaw)
    a14 = np.dot(a34,np.dot(a23,a12))
    return a14

def position(r,a12,a14):
    pre_rT = np.ones(length)*H   
    rT =  np.array([pre_rT, np.zeros_like(pre_rT), np.zeros_like(pre_rT)])
    a21 = a12.transpose()
    pre_rS = np.ones(length)*(-L)
    pre_rS2 = np.array([np.zeros_like(pre_rS), np.zeros_like(pre_rS), pre_rS])
    rS =np.dot( a21,pre_rS2)
    
    a41 = a14.transpose()
    rB = np.dot(a41, np.array([r, np.zeros_like(r), np.zeros_like(r)]))

    return rT + rS + rB
def constant_wind(V_hub):
    V0_array = np.ones(length)*V_hub
    return np.array([np.zeros(length),np.zeros(length),V0_array])

def wind_shear(x, V_hub, H, nu):
    v_shear = np.array([np.zeros_like(x),np.zeros_like(x),V_hub*(x/H)**nu])
    return v_shear
    
def tower_speed(V0, coord):
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

def pre_interpolate(airfoils):
    #interpolate the values to the different thicknesses
    clthick = [] #initialise
    cdthick = []
    for foil in airfoils: # k indicates the airfoil
        clthick.append(interp1d(foil[:,0],foil[:,1], kind="linear", bounds_error=False, fill_value="extrapolate"))
        cdthick.append(interp1d(foil[:,0], foil[:,2], kind="linear", bounds_error=False, fill_value="extrapolate"))
    return clthick, cdthick

def interpolate(alpha, clthick, cdthick, thicknesses):
    """interpolate to find the lift and drag coefficients"""
    cl = np.zeros(length)
    cd = np.zeros(length)
    for idx, a in enumerate(alpha):
        cl_temps = np.array([f(a) for f in clthick])   # shape (6,)
        cd_temps = np.array([f(a) for f in cdthick]) 
        
        #then interpolate to the actual thickness
        thick_prof=np.array([100,60,48,36,30.1,24.1])
        order = np.argsort(thick_prof)           # ascending order
        thick_sorted = thick_prof[order]
        clift=interp1d(thick_sorted[:],cl_temps[:])
        cdrag=interp1d(thick_sorted[:],cd_temps[:])
        cl[idx] = clift(thicknesses[idx])
        cd[idx] = cdrag(thicknesses[idx])
    return {"Cl": cl, "Cd": cd}


def wind_velocity(theta_cone,theta_yaw,theta_tilt,omega,dt,N, x, V_hub, H, nu):
    thetas = np.zeros((N,B))
    thetas[0] = 0,2*np.pi/B, 4*np.pi/3
    
    shear_vel = np.zeros((B,N,3, length))
    velocities = np.zeros((B,N,3, length))
    velocities_in4 = np.zeros((B,N,3, length))
    p_y = np.zeros((B,N,length))
    p_z = np.zeros((B,N,length))

    r_array = np.zeros((B,N,3,length))

    W_qs_y_old = np.zeros(length)
    W_qs_z_old = np.zeros(length)

    f_g = np.zeros(length)
    
    for i in range(0,N):
        time = i*dt
        if i<N-1:
            thetas[i+1] = np.array(np.linspace(thetas[i,0]+omega*dt, omega*dt+(B-1)/B*2*np.pi, B))

        for j in range(B):
            theta = thetas[i,j]
            a23 = matrix_a23(theta) #update matrix for each blade
            a14= matrix_a14(theta_cone, theta_tilt, theta_yaw, a23)
           
            
            r_array[j,i] = position(radii,matrices_notime(theta_cone, theta_tilt, theta_yaw)[0], a14)
            
            shear_vel[j,i] = wind_shear(r_array[j,i,0], V_hub, H, nu)

            velocities[j,i] = constant_wind(V0)

            if Shear: 
                velocities[j,i] = wind_shear(r_array[j,i,0], V_hub, H, nu)

            if Tower:
                velocities[j,i] = tower_speed(velocities[j,i], r_array[j,i])
                
            velocities_in4[j,i] = np.dot(a14,velocities[j,i])

            V0_y = velocities_in4[j,i,1]
            V0_z = velocities_in4[j,i,2]

            V_rel_y = V0_y + W_qs_y_old - omega*radii*np.cos(theta_cone)
            V_rel_z = V0_z + W_qs_z_old
            V_rel = np.sqrt(V_rel_y**2+V_rel_z**2)
            phi = np.arctan((V_rel_z/(-V_rel_y)))
            pitch = np.ones(length)*theta_pitch[j]
            alpha = np.rad2deg(phi)-(betas+pitch)
            
            coeff = interpolate(alpha, clthick, cdthick, thicknesses) 
            Cl = coeff['Cl']
            Cd = coeff['Cd']
            l = 0.5*rho*V_rel**2*chords*Cl
            d = 0.5*rho*V_rel**2*chords*Cd
            p_z[j,i] = l*np.cos(phi)+d*np.sin(phi)
            p_y[j,i] = l*np.sin(phi)-d*np.cos(phi)

            a = (-W_qs_z_old/V_hub)
            for idx,a_loop in enumerate(a):
                if a_loop<=1/3:
                    f_g[idx] = 1
                else:
                    f_g[idx] = 1/4*(5-3*a_loop)
            F = (2/np.pi)*(np.arccos(np.exp((-B*(R-radii))/(2*radii*np.sin(np.abs(phi))))))

            Norm = np.sqrt(V0_y**2+(V0_z+f_g*W_qs_z_old)**2)
            W_qs_z = (-B*l*np.cos(phi)/(4*np.pi*rho*radii*F*Norm))
            W_qs_y = (-B*l*np.sin(phi)/(4*np.pi*rho*radii*F*Norm))

            W_qs_y_old = W_qs_y
            W_qs_z_old = W_qs_z
    p_y[:,:,-1] = 0
    p_z[:,:,-1] = 0
        
    return thetas, r_array, velocities_in4, p_y, p_z

labels = ['No yaw', 'Yaw = 20°']
colors = ['tab:red', 'tab:cyan']
j = 0
Vy = np.zeros((2,N, length))
Vz = np.zeros((2,N, length))
clthick, cdthick = pre_interpolate(airfoils) 
for theta_yaw in (0,np.deg2rad(20)):
    angles, positions, speeds, pys, pzs = wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, x_blade, V_hub, H, nu)
    x_array = positions[0,:,0]
    y_array = positions[0,:,1]
    Vy[j] = speeds[0,:,1]
    Vz[j] = speeds[0,:,2]
    plt.plot(y_array, x_array, label=labels[j], color=colors[j])
    j=j+1
    
    
    
    
        
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

theta_yaw =np.deg2rad(20)
angles, positions, speeds, pys, pzs = wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, x_blade, V_hub, H, nu)
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


plt.plot(radii,pys[0,-2],label='py')
plt.plot(radii,pzs[0,-2],label='pz')
plt.legend()
plt.show()