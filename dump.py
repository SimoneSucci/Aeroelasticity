labels = ['No yaw', 'Yaw = 20°']
colors = ['tab:red', 'tab:cyan']
Vy = np.zeros((2,N, length))
Vz = np.zeros((2,N, length))

clthick, cdthick, fs_stat_thick, cl_inv_thick, cl_fs_thick = pre_interpolate(airfoils) 
time, angles, positions, speeds, pys, pzs, P, T1, T2, T3, T, Wy, Wz = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
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
time, angles, positions, speeds, pys, pzs, P, T1,T2,T3,T, Wy, Wz = simulate_wind_velocity(theta_cone, theta_yaw, theta_tilt,omega, dt, N, V_hub)
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

Vy_result1 = speeds[0,:,1, 5]
Vy_result2 = speeds[1,:,1, 5]
Vy_result3 = speeds[2,:,1, 5]
Vz_result1 = speeds[0,:,2, 5]
Vz_result2 = speeds[1,:,2, 5]
Vz_result3 = speeds[2,:,2, 5]
blade1 = angles[:,0]
blade2 = angles[:,1]
blade3 = angles[:,2]
plt.plot(blade1, Vy_result1, label = '$V_y$', color='tab:red')
plt.plot(blade1, Vz_result1, label = '$V_z$', color = 'tab:cyan')
plt.legend()
#plt.xlim(4*np.pi/3,4*np.pi/3+2*np.pi)
plt.grid()
plt.show()

plt.plot(blade2, Vy_result2, label = '$V_y$', color='tab:red')
plt.plot(blade2, Vz_result2, label = '$V_z$', color = 'tab:cyan')
plt.legend()
#plt.xlim(4*np.pi/3,4*np.pi/3+2*np.pi)
plt.grid()
plt.show()

plt.plot(blade3, Vy_result3, label = '$V_y$', color='tab:red')
plt.plot(blade3, Vz_result3, label = '$V_z$', color = 'tab:cyan')
plt.legend()
#plt.xlim(4*np.pi/3,4*np.pi/3+2*np.pi)
plt.grid()
plt.show()



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