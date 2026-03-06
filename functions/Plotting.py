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
    ax.set_title(f'Distribution of loads on blade 1 for t={round(t_array[t_idx], 1)} s')

    plt.show()

    return fig, ax

def plot_load_history(t_array: np.ndarray, pz: np.ndarray, radii: np.ndarray, r: float, t_start):
    """ Takes an array of pz and py and plots them over radial position for specific blade and time."""
    r_idx = np.where(np.round(radii, 2)==r)
    pz_plot = np.squeeze(pz[0,:,r_idx])

    fig = plt.figure(num = 1, figsize=(9, 4))
    ax = plt.axes()
    
    ax.plot(t_array[t_start:], pz_plot[t_start:])
    ax.set_ylabel('Normal load $p_z$ [N/m]')
    ax.set_xlabel('Time [s]')
    ax.set_title(f'Time history of $p_z$ on blade 1 for r={r} m')

    plt.show()

    return fig, ax

def plot_PT_history(t_array: np.ndarray, P: np.ndarray, T: np.ndarray, t_start: float, Dynamic_wake, P_wake = None, T_wake = None, each_blade = False, P_ashes = None, T_ashes = None, t_ashes = None, T1 = None, T2 = None, T3 = None, only_one = False):
    """ Takes an array of P and T and plots them over time."""
    P_plot = P/10**6
    T_plot = T/10**3
    if Dynamic_wake:
        T_wake_plot = T_wake/10**3
        P_wake_plot = P_wake/10**6

    fig, axs = plt.subplots(2,1, figsize=(9, 6))
    if Dynamic_wake:
        axs[0].plot(t_array[t_start:], P_wake_plot[t_start:], color = 'c', label = 'Power with dynamic wake')
    axs[0].plot(t_array[t_start:], P_plot[t_start:],'--', label = 'Power')
    if each_blade:
        axs[0].plot(t_ashes, P_ashes, label = 'Ashes')
    axs[0].set_ylabel('Power [MW]')
    axs[0].set_xlabel('Time [s]')
    axs[0].grid()
    #axs[0].set_title('Time history of total power')
    axs[0].legend()
    #axs[0].set_ylim(bottom=0)
    
    if each_blade:
        T1_plot = T1/10**3
        T2_plot = T2/10**3
        T3_plot = T3/10**3
        axs[1].plot(t_array[t_start:], T1_plot[t_start:], label = 'Blade 1')
        axs[1].plot(t_array[t_start:], T2_plot[t_start:], label = 'Blade 2')
        axs[1].plot(t_array[t_start:], T3_plot[t_start:], label = 'Blade 3')
        axs[1].plot(t_array[t_start:], T_plot[t_start:], label = 'Total')
        axs[1].plot(t_ashes, T_ashes, label = 'Total Ashes')
        
        axs[1].set_title('Time history of thrusts')
        axs[1].legend(loc='center right')
    elif only_one:
        T1_plot = T1/10**3
        axs[1].plot(t_array[t_start:], T1_plot[t_start:])
        axs[1].set_title('Time history of thrust on blade 1')
    else: 
        if Dynamic_wake:
            axs[1].plot(t_array[t_start:], T_wake_plot[t_start:], color = 'c',label = 'Thrust with dynamic wake')
        axs[1].plot(t_array[t_start:], T_plot[t_start:], '--', label = 'Thrust')
        #axs[1].set_title('Time history of total thrust')
        axs[1].legend()
        axs[1].set_ylim(bottom=0)
        
    axs[1].grid()
    axs[1].set_ylabel('Thrust kN]')
    axs[1].set_xlabel('Time [s]')

    plt.tight_layout()
    
    plt.show()

    return fig, axs

def plot_induced_wind(t_array: np.ndarray, Wy: np.ndarray, Wz: np.ndarray, radii: np.ndarray, r: float, Dynamic_wake, Wy_wake=None, Wz_wake=None):
    """ Takes an array of Wz and Wy and plots them over time for specific blade and blade position."""
    r_idx = np.where(np.round(radii, 2)==r)
    Wz_plot = np.squeeze(Wz[:, 0, r_idx])
    Wy_plot = np.squeeze(Wy[:, 0, r_idx])
    if Dynamic_wake:
        Wy_wake_plot = np.squeeze(Wy_wake[:, 0, r_idx])
        Wz_wake_plot = np.squeeze(Wz_wake[:, 0, r_idx])

    fig, axs = plt.subplots(2,1, figsize=(9, 6))
    
    if Dynamic_wake:
        axs[0].plot(t_array, Wy_wake_plot, color = 'c', label = '$W_y$ with dynamic wake')
    axs[0].plot(t_array, Wy_plot, '--', label = '$W_y$')
    axs[0].set_ylabel('$W_y$ [m/s]')
    axs[0].set_xlabel('Time [s]')
    axs[0].grid()
    axs[0].legend()
    #axs[0].set_ylim(top=0)
    #axs[0].set_title(f'Time history of induced wind $W_y$ on blade 1 for r={r} m')

    if Dynamic_wake:
        axs[1].plot(t_array, Wz_wake_plot, color = 'c', label = '$W_z$ with dynamic wake')
    axs[1].plot(t_array, Wz_plot, '--', label = '$W_z$')
    axs[1].set_ylabel('$W_z$ [m/s]')
    axs[1].set_xlabel('Time [s]')
    axs[1].grid()
    axs[1].legend()
    #axs[1].set_title(f'Time history of induced wind $W_z$ on blade 1 for r={r} m')
    #axs[1].set_ylim(top=0)

    plt.tight_layout()
    plt.show()

    return fig, axs

def plot_PSDs(dt: float, pz: np.ndarray, T1: np.ndarray, radii: np.ndarray, r: float, t_start: float, t_end:float, omega):
    """ Takes an array of pz and T1 over time and plots their PSD."""
    r_idx = np.where(np.round(radii, 2)==r)
    pz_plot = np.squeeze(pz[0, :, r_idx])
    fs = 1/dt

    f_pz, PSD_pz = signal.welch(pz_plot[t_start:t_end], fs, nperseg = 500)
    f_T1, PSD_T1 = signal.welch(T1[t_start:t_end], fs, nperseg=500)

    fig, axs = plt.subplots(2,1, figsize=(9, 4))
    
    axs[0].plot(2*np.pi/omega*f_pz, PSD_pz)
    axs[0].set_ylabel('PSD($p_z$)')
    axs[0].set_xlabel('Frequency [Hz]')
    axs[0].set_title('PSD of Normal Load')
    
    axs[1].plot(2*np.pi/omega*f_T1, PSD_T1)
    axs[1].set_ylabel('PSD(T)')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_title('PSD of Thrust on blade 1')

    plt.tight_layout()
    plt.show()

    return fig, axs

def plot_PSD_Q2(dt: float, P: np.ndarray, T1: np.ndarray, T: np.ndarray, t_start: float, t_end:float, omega):
    """ Takes an array of P , T1, and T over time and creates two subplots with their PSDs."""
    fs = 1/dt

    f_P, PSD_P = signal.welch(P[t_start:t_end], fs, nperseg = 500)
    f_T1, PSD_T1 = signal.welch(T1[t_start:t_end], fs, nperseg=500)
    f_T, PSD_T = signal.welch(T[t_start:t_end], fs, nperseg=500)


    fig, axs = plt.subplots(2,1, figsize=(9, 6))
    
    axs[0].semilogy(2*np.pi/omega*f_P, PSD_P)
    axs[0].set_ylabel('PSD of power')
    axs[0].set_xlabel(r"$\frac{2 \pi f}{\omega}$ [-]")
    axs[0].grid()
    #axs[0].set_title('PSD of Normal Load')
    
    axs[1].semilogy(2*np.pi/omega*f_T1, PSD_T1, 'tab:green', label = 'Thrust on 1 blade')
    axs[1].semilogy(2*np.pi/omega*f_T, PSD_T, 'tab:orange', label= 'Total thrust')
    axs[1].set_ylabel('PSD of thrusts')
    axs[1].set_xlabel(r"$\frac{2 \pi f}{\omega}$ [-]")
    axs[1].grid()
    axs[1].legend()
    #axs[1].set_title('PSD of Thrust on blade 1')

    plt.tight_layout()
    plt.show()

    return fig, axs