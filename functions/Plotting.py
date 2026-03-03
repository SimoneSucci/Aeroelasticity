import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def plot_loads_distribution(r_array, py, pz, t_idx, t_array):

    pz_plot = pz[0,t_idx,:]
    py_plot = py[0,t_idx,:]

    fig = plt.figure(num = 1, figsize=(9, 4))
    ax = plt.axes()
    
    ax.plot(r_array, py_plot, label='Tangential: $p_y$')
    ax.plot(r_array, pz_plot, label = 'Normal $p_z$')

    ax.set_ylabel('Loads [N/m]')
    ax.set_xlabel('Radial position [m]')
    ax.set_title(f'Distribution of loads on blade 1 for t={t_array[t_idx]} s')

    plt.show()

    return fig, ax

def plot_load_history(t_array: np.ndarray, pz: np.ndarray, radii: np.ndarray, r: float):
    """ Takes an array of pz and py and plots them over radial position for specific blade and time."""
    r_idx = np.where(np.round(radii, 2)==r)
    pz_plot = pz[0,:,r_idx]

    fig = plt.figure(num = 1, figsize=(9, 4))
    ax = plt.axes()
    
    ax.plot(t_array, pz_plot)
    ax.set_ylabel('Normal load $p_z$ [N/m]')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'Time history of $p_z$ on blade 1 for r={r} m')

    plt.show()

    return fig, ax

def plot_PT_history(t_array: np.ndarray, P: np.ndarray, T: np.ndarray, each_blade = False, T1 = None, T2 = None, T3 = None, only_one = False):
    """ Takes an array of P and T and plots them over time."""

    fig, axs = plt.subplots(2,1, figsize=(9, 4))
    
    axs[0].plot(t_array, P)
    axs[0].set_ylabel('Power [W]')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_title('Time history of total power')
    
    if each_blade:
        axs[1].plot(t_array, T1, label = 'Blade 1')
        axs[1].plot(t_array, T2, label = 'Blade 2')
        axs[1].plot(t_array, T3, label = 'Blade 3')
        axs[1].plot(t_array, T, label = 'Total')
        axs[1].set_title('Time history of thrusts')
    elif only_one:
        axs[1].plot(t_array, T1)
        axs[1].set_title('Time history of thrust on blade 1')
    else: 
        axs[1].plot(t_array, T)
        axs[1].set_title('Time history of total thrust')

    axs[1].set_ylabel('Thrust [N]')
    axs[1].set_xlabel('Time [s]')
    
    if each_blade:
        axs[1].legend()

    plt.tight_layout()
    
    plt.show()

    return fig, axs

def plot_induced_wind(t_array: np.ndarray, Wy: np.ndarray, Wz: np.ndarray, radii: np.ndarray, r: float):
    """ Takes an array of Wz and Wy and plots them over time for specific blade and blade position."""
    r_idx = np.where(np.round(radii, 2)==r)
    Wz_plot = Wz[:, 0, r_idx]
    Wy_plot = Wy[:, 0, r_idx]

    fig = plt.figure(num = 1, figsize=(9, 4))
    ax = plt.axes()
    
    ax.plot(t_array, Wz_plot, label = '$W_z$')
    ax.plot(t_array, Wy_plot, label = '$W_y$')
    ax.set_ylabel('Induced wind [m/s]')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'Time history of induced wind on blade 1 for r={r} m')

    plt.show()

    return fig, ax

def plot_PSDs(dt: float, pz: np.ndarray, T1: np.ndarray, radii: np.ndarray, r: float):
    """ Takes an array of Wz and Wy and plots them over time for specific blade and blade position."""
    r_idx = np.where(np.round(radii, 2)==r)
    pz_plot = pz[:, 0, r_idx]
    fs = 1/dt

    f_pz, PSD_pz = signal.welch(pz, fs, nperseg=1024)
    f_T1, PSD_T1 = signal.welch(T1, fs, nperseg=1024)

    fig, axs = plt.subplots(2,1, figsize=(9, 4))
    
    axs[0].plot(f_pz, PSD_pz)
    axs[0].set_ylabel('PSD($p_z$)')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_title('PSD of Normal Load')
    
    axs[1].plot(f_T1, PSD_T1)
    axs[1].set_ylabel('PSD(T)')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_title('PSD of Thrust on blade 1')

    plt.tight_layout()
    plt.show()

    return fig, axs