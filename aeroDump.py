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

Dynamic_stall = False

def load_airfoils(thickness1_file: str)-> List:
    """Loads the airfoil data: CT and Cd for each airfoil shape. 
    All airfoils are then collected in a nested list."""

    airfoil = np.loadtxt(thickness1_file)    

    return airfoil

airfoil = load_airfoils(DATA_DIR / 'FFA-W3-241_ds.txt')





#constant

A = 0.2 #m
omega = 3 #rad/s
alpha0 = np.deg2rad([5, 10, 15, 20]) #rad
rho = 1.225 # kg/m*3
chord = 1 #m

T = 2*np.pi/omega
N_dt = 20
dt = T/N_dt
N_cycles = 20
N = N_cycles*N_dt
time= np.linspace(0,N*dt,N)
theta = np.deg2rad(np.linspace(0,180,181))

V0 = np.ones(len(theta))*5 #m/s

W = np.empty((len(alpha0),len(theta)))
W_accu = np.empty((N,len(alpha0),len(theta)))
alpha= np.empty((N,len(alpha0),len(theta)))
cl= np.empty((N,len(alpha0),len(theta)))
cd= np.empty((N,len(alpha0),len(theta)))
Fx = np.zeros((N,len(alpha0),len(theta)))

for j , a0  in enumerate(alpha0):

    fs_old= np.zeros(len(theta))

    for i in range(0,N):

        #time[i] = i*dt

        x = A * np.sin(omega*time[i])
        x_dot = omega*A * np.cos(omega*time[i])
        
        V= np.array ([V0*np.cos(a0)+x_dot*np.cos(theta),
                    V0*np.sin(a0)+x_dot*np.sin(theta)])
        
        Vrel = np.sqrt(V[0]**2+V[1]**2)

        alpha[i,j] = (np.arctan(V[1]/V[0]))

        if Dynamic_stall:
            tau = 4*chord/Vrel
            fs_stat = np.interp (np.rad2deg(alpha[i,j]),airfoil[:,0],airfoil[:,4])
            Cl_inv = np.interp (np.rad2deg(alpha[i,j]),airfoil[:,0],airfoil[:,5])
            Cl_fs = np.interp (np.rad2deg(alpha[i,j]),airfoil[:,0],airfoil[:,6])
            fs = fs_stat+(fs_old-fs_stat)*np.exp(-dt/tau)
            cl[i,j] = fs*Cl_inv+(1-fs)*Cl_fs
            fs_old = fs
        else:
            cl[i,j] = np.interp (np.rad2deg(alpha[i,j]),airfoil[:,0],airfoil[:,1])

        cd[i,j] = np.interp (np.rad2deg(alpha[i,j]),airfoil[:,0],airfoil[:,2])
        

        l = 0.5*rho*Vrel**2*chord*cl[i,j]
        d = 0.5*rho*Vrel**2*chord*cd[i,j]


        Fx[i,j] = l * np.sin(alpha[i,j] -theta) - d * np.cos(alpha[i,j] -theta)
        W_accu[i,j] = A * omega * np.trapz (Fx[:,j]*np.cos(omega*time[:, None]),time, axis=0)

    W[j] = A * omega * np.trapz (Fx[:,j]*np.cos(omega*time[:, None]),time, axis=0)/N_cycles

alpha0=np.round(np.rad2deg(alpha0),0)
'''plt.figure()
plt.plot(np.rad2deg(theta),W[0],label=f'alpha:{alpha0[0]}')
plt.plot(np.rad2deg(theta),W[1],label=f'alpha:{alpha0[1]}')
plt.plot(np.rad2deg(theta),W[2],label=f'alpha:{alpha0[2]}')
plt.plot(np.rad2deg(theta),W[3],label=f'alpha:{alpha0[3]}')
plt.axhline(y=0, color='r', linestyle='--', label='W = 0')
plt.legend()
plt.figure()
plt.plot(time,W_accu[:,1,0])
plt.figure()
plt.plot(time,Fx[:,1,0])
plt.figure()
plt.plot(time,np.rad2deg(alpha[:,1,0]))
'''
plt.figure()
plt.plot(np.rad2deg(alpha[:,:,89]),cl[:,:,89],label='cl')

plt.plot(airfoil[50:70,0],airfoil[50:70,1],linestyle='--')
plt.legend()
plt.show()



