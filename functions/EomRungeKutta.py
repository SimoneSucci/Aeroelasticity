import numpy as np


def calculate_g(DOF,y,y_d, M, k_t, m, I, radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes):
    u_y_1f, u_z_1f, u_y_1e, u_z_1e, u_y_2f, u_z_2f = modes
    omega_1f, omega_1e, omega_2f = omegas_modes 



    M11 = M + 3*np.trapz(m, radii)
    M12 = 0
    M13 = np.trapz(m*u_z_1f, radii)
    M14 = np.trapz(m*u_z_1e, radii)
    M15 = np.trapz(m*u_z_2f, radii)
    M21 = 0
    M22 = I
    M23 = np.trapz(m*radii*u_y_1f, radii)
    M24 = np.trapz(m*radii*u_y_1e, radii)
    M25 = np.trapz(m*radii*u_y_2f, radii)
    M31 = M13
    M32 = M23
    GM1 = np.trapz(u_y_1f*m*u_y_1f, radii) + np.trapz(u_z_1f*m*u_z_1f, radii)
    M33 = GM1
    M34 = 0
    M35 = 0
    M41 = M14
    M42 = M24
    M43 = 0
    GM2 = np.trapz(u_y_1e*m*u_y_1e, radii) + np.trapz(u_z_1e*m*u_z_1e, radii)
    M44 = GM2
    M45 = 0
    M51 = M15
    M52 = M25
    M53 = 0
    M54 = 0
    GM3 = np.trapz(u_y_2f*m*u_y_2f, radii) + np.trapz(u_z_2f*m*u_z_2f, radii)
    M55 = GM3

    if DOF ==5:
        M_matrix= np.array( [[M11,M12,M13,M14,M15],
                            [M21,M22,M23,M24,M25],
                            [M31,M32,M33,M34,M35],
                            [M41,M42,M43,M44,M45],
                            [M51,M52,M53,M54,M55]
                            ])
        
        K_matrix = np.array([[k_t,0,0,0,0],
                            [0,0,0,0,0],
                            [0,0,omega_1f**2*GM1,0,0],
                            [0,0,0,omega_1e**2*GM2,0],
                            [0,0,0,0,omega_2f**2*GM3]])
        
        GF = np.array([[Thrust],
                    [Torque - MG],
                    [np.trapz(p_y[0]*u_y_1f, radii) + np.trapz(p_z[0]*u_z_1f, radii)],
                    [np.trapz(p_y[0] *u_y_1e, radii) + np.trapz(p_z[0]*u_z_1e, radii)],
                    [np.trapz(p_y[0] *u_y_2f, radii) + np.trapz(p_z[0]*u_z_2f, radii)]
                    ])
    elif DOF==11:
        M_matrix= np.array( [[M11,M12,M13,M14,M15,M13,M14,M15,M13,M14,M15],
                            [M21,M22,M23,M24,M25,M23,M24,M25,M23,M24,M25],
                            [M31,M32,M33,M34,M35,0,0,0,0,0,0],
                            [M41,M42,M43,M44,M45,0,0,0,0,0,0],
                            [M51,M52,M53,M54,M55,0,0,0,0,0,0],
                            [M31,M32,0,0,0,GM1,0,0,0,0,0],
                            [M41,M42,0,0,0,0,GM2,0,0,0,0],
                            [M51,M52,0,0,0,0,0,GM3,0,0,0],
                            [M31,M32,0,0,0,0,0,0,GM1,0,0],
                            [M41,M42,0,0,0,0,0,0,0,GM2,0],
                            [M51,M52,0,0,0,0,0,0,0,0,GM3]
                            ])
        
        K_matrix = np.array([[k_t,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,omega_1f**2*GM1,0,0,0,0,0,0,0,0],
                            [0,0,0,omega_1e**2*GM2,0,0,0,0,0,0,0],
                            [0,0,0,0,omega_2f**2*GM3,0,0,0,0,0,0],
                            [0,0,0,0,0,omega_1f**2*GM1,0,0,0,0,0],
                            [0,0,0,0,0,0,omega_1e**2*GM2,0,0,0,0],
                            [0,0,0,0,0,0,0,omega_2f**2*GM3,0,0,0],
                            [0,0,0,0,0,0,0,0,omega_1f**2*GM1,0,0],
                            [0,0,0,0,0,0,0,0,0,omega_1e**2*GM2,0],
                            [0,0,0,0,0,0,0,0,0,0,omega_2f**2*GM3]])
        
        GF = np.array([[Thrust],
                    [Torque - MG],
                    [np.trapz(p_y[0]*u_y_1f, radii) + np.trapz(p_z[0]*u_z_1f, radii)],
                    [np.trapz(p_y[0]*u_y_1e, radii) + np.trapz(p_z[0]*u_z_1e, radii)],
                    [np.trapz(p_y[0]*u_y_2f, radii) + np.trapz(p_z[0]*u_z_2f, radii)],
                    [np.trapz(p_y[1]*u_y_1f, radii) + np.trapz(p_z[1]*u_z_1f, radii)],
                    [np.trapz(p_y[1]*u_y_1e, radii) + np.trapz(p_z[1]*u_z_1e, radii)],
                    [np.trapz(p_y[1]*u_y_2f, radii) + np.trapz(p_z[1]*u_z_2f, radii)],
                    [np.trapz(p_y[2]*u_y_1f, radii) + np.trapz(p_z[2]*u_z_1f, radii)],
                    [np.trapz(p_y[2]*u_y_1e, radii) + np.trapz(p_z[2]*u_z_1e, radii)],
                    [np.trapz(p_y[2]*u_y_2f, radii) + np.trapz(p_z[2]*u_z_2f, radii)]
                    ])
    
    g = np.linalg.solve(M_matrix,(GF-K_matrix @ y))
   
    #g = g.flatten()
   
    return g

def calculate_g_3(y,y_d, M, k_t, m, I, radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes):
    u_y_1f, u_z_1f, u_y_1e, u_z_1e, u_y_2f, u_z_2f = modes
    omega_1f, omega_1e, omega_2f = omegas_modes 


    
    GM1 = np.trapz(u_y_1f*m*u_y_1f, radii) + np.trapz(u_z_1f*m*u_z_1f, radii)
    M33 = GM1
    M34 = 0
    M35 = 0
    M43 = 0
    GM2 = np.trapz(u_y_1e*m*u_y_1e, radii) + np.trapz(u_z_1e*m*u_z_1e, radii)
    M44 = GM2
    M45 = 0
    M53 = 0
    M54 = 0
    GM3 = np.trapz(u_y_2f*m*u_y_2f, radii) + np.trapz(u_z_2f*m*u_z_2f, radii)
    M55 = GM3

    M_matrix= np.array( [
                         [M33,M34,M35],
                         [M43,M44,M45],
                         [M53,M54,M55]
                          ])
    
    K_matrix = np.array([
                         [omega_1f**2*GM1,0,0],
                         [0,omega_1e**2*GM2,0],
                         [0,0,omega_2f**2*GM3]])
    
    GF = np.array([
                  [np.trapz(p_y*u_y_1f, radii) + np.trapz(p_z*u_z_1f, radii)],
                  [np.trapz(p_y*u_y_1e, radii) + np.trapz(p_z*u_z_1e, radii)],
                  [np.trapz(p_y*u_y_2f, radii) + np.trapz(p_z*u_z_2f, radii)]
                  ])
    
    g = np.linalg.solve(M_matrix,(GF-K_matrix @ y))
   
    #g = g.flatten()
   
    return g

def rungeKutta(DOF,dt,y,y_d,y_dd,M,k_t,m,I, radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes):
    y = y.reshape(DOF,1)
    y_d = y_d.reshape(DOF,1)
    y_dd = y_dd.reshape(DOF,1)

    A = dt*y_dd/2
    b = dt *(y_d+0.5*A)/2

    g2 = calculate_g(DOF,y+b,y_d+A,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )
    B = dt*g2/2
    g3 = calculate_g(DOF,y+b,y_d+B,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )
    C = dt*g3/2 

    d = dt*(y_d+C)
    g4 = calculate_g(DOF,y+d, y_d+2*C,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes)
    D = dt*g4/2

    y_new = y + dt*(y_d+(A+B+C)/3)
    y_d_new = y_d + ((A+2*B+2*C+D)/3)
    y_dd_new = calculate_g(DOF,y_new, y_d_new,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )

    return y_new.reshape(DOF,), y_d_new.reshape(DOF,), y_dd_new.reshape(DOF,)
 
def rungeKutta_3(dt,y,y_d,y_dd,M,k_t,m,I, radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes):
    y = y.reshape(3,1)
    y_d = y_d.reshape(3,1)
    y_dd = y_dd.reshape(3,1)

    A = dt*y_dd/2
    b = dt *(y_d+0.5*A)/2

    g2 = calculate_g_3(y+b,y_d+A,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )
    B = dt*g2/2
    g3 = calculate_g_3(y+b,y_d+B,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )
    C = dt*g3/2 

    d = dt*(y_d+C)
    g4 = calculate_g_3(y+d, y_d+2*C,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes)
    D = dt*g4/2

    y_new = y + dt*(y_d+(A+B+C)/3)
    y_d_new = y_d + ((A+2*B+2*C+D)/3)
    y_dd_new = calculate_g_3(y_new, y_d_new,M,k_t,m,I,radii, Thrust, Torque, MG, p_y, p_z, omegas_modes, modes )

    return y_new.reshape(3,), y_d_new.reshape(3,), y_dd_new.reshape(3,)
 


