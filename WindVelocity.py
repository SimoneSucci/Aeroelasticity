#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:55:12 2026

@author: ombeline
"""

import numpy as np
import matplotlib.pyplot as plt

B = 3

H = 119 # hub height
L = 7.1 #shaft
R = 89.15 #blade radius
theta_tilt = np.deg2rad(-5)
theta_cone = 0
theta_yaw = 0
x_blade = 70

omega = 0.62 #angular velocity
dt = 0.15 #time step
N = 500
a_tower = 3.32
V0 = 10

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

def matrix_a14(theta_cone, theta_tilt,theta_yaw, theta_blade, a23):
    a12, a34 = matrices_notime(theta_cone, theta_tilt, theta_yaw)
    a14 = np.dot(a34,np.dot(a23,a12))
    return a14

def position(a12,a14):
    rT = np.array([H,0,0])
    
    a21 = a12.transpose()
    rS =np.dot( a21,np.array([0,0,-L]))
    
    a41 = a14.transpose()
    rB = np.dot(a41, np.array([x_blade, 0, 0]))
    return rT+rS+rB

def wind_shear(x, V_hub, H, nu):
    v_shear = np.array([0,0,V_hub*(x/H)**nu])
    return v_shear
    
def tower_speed(x, V0, coord):
    if x<H:
        a = a_tower
    else:
        a=0
    y, z = coord[1], coord[2]
    r = np.sqrt(y**2+z**2)
    Vr = z/r*V0*(1-(a/r)**2)
    Vt = y/r*V0*(1+(a/r)**2)
    
    Vx = 0
    Vy = y/r*Vr-z/r*Vt
    Vz = z/r*Vr+y/r*Vt

    return Vx, Vy, Vz
    
def wind_velocity(theta_cone,theta_yaw,theta_tilt,omega,dt,N, x, V_hub, H, nu):
    theta1 = np.zeros(N)
    theta2 = np.zeros(N)
    theta3 = np.zeros(N)
    theta2[0] = 2*np.pi/3
    theta3[0] = 4*np.pi/3
    thetas = np.zeros((N,B))
    
    shear_vel = np.zeros((B,N,3))
    speed_in1_shear = np.zeros((B,N,3))
    speed_in1_noshear = np.zeros((B,N,3))
    speed_in4_shear = np.zeros((B,N,3))
    speed_in4_noshear = np.zeros((B,N,3))
    r_array = np.zeros((B,N,3))
    
    for i in range(0,N-1):
        time = i*dt
        theta1[i+1] = theta1[i]+omega*dt
        theta2[i+1] = theta1[i+1] + 2*np.pi/3 
        theta3[i+1] = theta1[i+1] + 4*np.pi/3
        thetas[i+1] = np.array([theta1[i+1],theta2[i+1],theta3[i+1]])
        
        for j in range(B):
            theta = thetas[i,j]
            a23 = matrix_a23(theta) #update matrix for each blade
            a14= matrix_a14(theta_cone, theta_tilt, theta_yaw, theta, a23)
           
            
            r_array[j,i] = position(matrices_notime(theta_cone, theta_tilt, theta_yaw)[0], a14)
             
            shear_vel[j,i] = wind_shear(r_array[j,i,0], V_hub, H, nu)
            speed_in1_noshear[j,i] = tower_speed(r_array[j,i,0], V0, np.dot(a14, r_array[j,i]))
    
            speed_in1_shear[j,i] = tower_speed(r_array[j,i,0], shear_vel[j,i,2], np.dot(a14, r_array[j,i]))
            speed_in4_shear[j,i] = np.dot(a14,speed_in1_shear[j,i])
            speed_in4_noshear[j,i] = np.dot(a14,speed_in1_noshear[j,i])

        
    return theta1, theta2, theta3, r_array, speed_in4_noshear, speed_in4_shear

labels = ['No yaw', 'Yaw = 20°']
j = 0
Vy = np.zeros((2,N))
Vz = np.zeros((2,N))
for theta_yaw in (0,np.deg2rad(20)):
    V_hub,nu = 10,0.2
    blade1, blade2, blade3, positions, speeds_noshear, speeds_shear = wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, x_blade, V_hub, H, nu)
    x_array = positions[0,:,0]
    y_array = positions[0,:,1]
    Vy[j] = speeds_shear[0,:,1]
    Vz[j] = speeds_shear[0,:,2]
    plt.plot(y_array, x_array, label=labels[j])
    j=j+1
    
    
        
plt.xlabel('y [m]')
plt.ylabel('x [m]')
plt.title('Trajectory of a point on Blade 1, for r = 70m')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()
plt.grid()
plt.show()

plt.plot(blade1, Vy[0], label = '$V_y$', color = 'tab:red')
plt.plot(blade1, Vz[0], label = '$V_z$', color = 'tab:cyan')
plt.legend()
plt.title('Tower and shear, No yaw')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()

plt.plot(blade1, Vy[1], label = '$V_y$', color = 'tab:red')
plt.plot(blade1, Vz[1], label = '$V_z$', color = 'tab:cyan')
plt.legend()
plt.title('Tower and shear, Yaw = 20°')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()

theta_yaw =np.deg2rad(20)
blade1, blade2, blade3, positions, speeds_noshear, speeds_shear = wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, x_blade, V_hub, H, nu)
Vy_result = speeds_noshear[0,:,1]
Vz_result = speeds_noshear[0,:,2]
plt.plot(blade1, Vy_result, label = '$V_y$', color='tab:red')
plt.plot(blade1, Vz_result, label = '$V_z$', color = 'tab:cyan')
plt.legend()
plt.title('Tower and no shear, yaw = 20°')
plt.xlabel('Azimuthal angle [rad]')
plt.ylabel('Velocity [m/s]')
plt.xlim(0,2*np.pi)
plt.grid()
plt.show()