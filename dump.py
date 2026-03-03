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