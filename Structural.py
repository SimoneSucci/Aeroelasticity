import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint 

#Fixing all the path so it works from any terminal
FILE_DIR = Path(__file__).parent  # directory where this file is located 

#sys.path.append(str(FUNCTION_DIR))
DATA_DIR = (FILE_DIR / 'data')

Dynamic_stall = True

# Input data

m =1 # kg
k = 61.7 # N/m
chord = 0.2 # m
span = 1 # m
rho = 1.225 # kg/m^3
v0 = 2 # m/s
alpha_g=0
params = [m, k, chord, span, rho, v0, alpha_g]

z0 = np.array([0.02, 0,0, alpha_g]) # Initial conditions for position, velocity, and f


# Import Airfoil 

def load_airfoils(thickness1_file: str)-> List:
    """Loads the airfoil data: CT and Cd for each airfoil shape. 
    All airfoils are then collected in a nested list."""

    airfoil = np.loadtxt(thickness1_file)    

    return airfoil

airfoil = load_airfoils(DATA_DIR / 'FFA-W3-241_ds.txt')


def calculate_dzdt(z: np.ndarray, t, params: List):
    z1, z2, f, alpha = z # Unpack variable array
    m, k, chord, span, rho, v0, alpha_g = params

    phi =np.arctan(z2/v0)
    alpha = np.deg2rad(alpha_g)+phi
    Vrel = np.sqrt(z2**2+v0**2)

    if Dynamic_stall:
        tau = 4*chord/Vrel
        f_stat = np.interp (np.rad2deg(alpha),airfoil[:,0],airfoil[:,4])
        Cl_inv = np.interp (np.rad2deg(alpha),airfoil[:,0],airfoil[:,5])
        Cl_fs = np.interp (np.rad2deg(alpha),airfoil[:,0],airfoil[:,6])

        cl = f*Cl_inv+(1-f)*Cl_fs
        df = (f_stat-f)/tau
        
    else:
        cl = np.interp (np.rad2deg(alpha),airfoil[:,0],airfoil[:,1])
        df = 0

    Fx = 0.5*rho*Vrel**2*chord*span*cl*np.cos(phi)

    dz1 = z2
    dz2 = (-k*z1-Fx)/m
    
    return [dz1, dz2, df, alpha]


t = np.linspace(0, 10, 1000)
sol1 = odeint(calculate_dzdt, z0, t, args=(params,))
print('stall off now')

Dynamic_stall = False

sol2 = odeint(calculate_dzdt, z0, t, args=(params,))

plt.plot(t, sol1[:,0], label='Dynamic Stall')
plt.plot(t, sol2[:,0], '--', label='No stall')
plt.xlabel('Time [s]')
plt.ylabel('Position x')
plt.legend()
plt.show()

plt.plot(t, sol1[:,1], label='Dynamic stall')
plt.plot(t, sol2[:,1], '--', label='No stall')
plt.xlabel('Time [s]')
plt.ylabel('Velocity x_dot')
plt.legend()
plt.show()

alpha_values1 = np.empty(len(t))
alpha_values2 = np.empty(len(t))
for i, time in enumerate(t):
    _, _, _, alpha1 = calculate_dzdt(sol1[i, :], time, params)
    _, _, _, alpha2 = calculate_dzdt(sol2[i, :], time, params)

    alpha_values1[i] = alpha1
    alpha_values2[i] = alpha2


plt.plot(t, alpha_values1, label='Dynamic stall')
plt.plot(t, alpha_values2, '--', label='No stall')
plt.xlabel('Time [s]')
plt.ylabel(' Alpha')
plt.legend()
plt.show()