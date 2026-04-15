import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# initial condition

x_0 = 0
x_d_0 = 0

theta_0 = 0
theta_d_0 = 0

Y0 = [x_0,x_d_0,theta_0,theta_d_0]

t = np.linspace (0,10,200)

#parameters

M = 1
m = 0.5
L = 2
g_grav = 9.81

#integration intialisation
dt = 0.1
N = 100
y_arr = np.zeros((N,2))
y_d_arr = np.zeros((N,2))
y_dd_arr = np.zeros((N,2))
time = np.zeros(N)

def calculate_dydt(y,t,M,m,L):

    x_d = y [1] 
    theta = y[2]
    theta_d = y[3]
    
    
    M_matrix= np.array( [ [M+m*L, -0.5*m*L**2*np.sin(theta)],

                          [ -0.5*m*L**2*np.sin(theta), m*L**3/3]
                          ])
    
    F = np.array([[0.5*m*L**2*theta_d**2*np.cos(theta)],
                  [0.5*m*L**2*g_grav*np.cos(theta)]
                  ])
    
    acc = np.linalg.inv (M_matrix) @ F
    
    x_dd, theta_dd = acc.flatten()

    dy_dt = [x_d, x_dd, theta_d, theta_dd]

    return dy_dt


def calculate_g(y,y_d,M,m,L):

    x = y[0]
    x_d = y_d[0]  

    theta = y[1]
    theta_d = y_d[1]
    
    M_matrix= np.array( [ [M+m*L, -0.5*m*L**2*np.sin(theta)],

                          [ -0.5*m*L**2*np.sin(theta), m*L**3/3]
                          ])
    
    F = np.array([[0.5*m*L**2*theta_d**2*np.cos(theta)],
                  [0.5*m*L**2*g_grav*np.cos(theta)]
                  ])
    
    g = np.linalg.inv (M_matrix) @ F
    g = g.flatten()

    return g

def rungeKutta(dt,t,y,y_d,y_dd,M,m,L):

    A = dt*y_dd/2
    b = dt *(y_d+0.5*A)/2

    g2 = calculate_g(y+b,y_d+A,M,m,L)
    B = dt*g2/2
    g3 = calculate_g(y+b,y_d+B,M,m,L)
    C = dt*g3/2

    d = dt*(y_d+C)
    g4 = calculate_g(y+d, y_d+2*C,M,m,L)
    D = dt*g4/2

    y_new = y + dt*(y_d+(A+B+C)/3)
    y_d_new = y_d + ((A+2*B+2*C+D)/3)
    y_dd_new = calculate_g(y_new, y_d_new,M,m,L)
    t_new = t + dt

    return y_new, y_d_new, y_dd_new, t_new



#odeint solution
sol = odeint (calculate_dydt,Y0,t, args=(M,m,L))

#Runge-Kutta solution
y_dd_arr[0] = calculate_g(y_arr[0], y_d_arr[0], M, m, L)
for i in range(N-1):
    y_arr[i+1], y_d_arr[i+1], y_dd_arr[i+1], time[i+1] = rungeKutta(dt, time[i], y_arr[i], y_d_arr[i], y_dd_arr[i], M, m, L)


# Plot and compare
plt.figure()
plt.plot(t,sol[:,0],label = 'x odeint')
plt.plot(t,sol[:,2],label = 'theta odeint')
plt.plot(time,y_arr[:,0],'--', label='x')
plt.plot(time,y_arr[:,1],'--', label='theta')
plt.xlabel('Time')
plt.ylabel('x, ${\\theta}$')
plt.legend()
plt.show()

plt.figure()
plt.plot(t,sol[:,1],label = 'x_d odeint')
plt.plot(t,sol[:,3],label = 'theta_d odeint')
plt.plot(time,y_d_arr[:,0], '--', label='x_d')
plt.plot(time,y_d_arr[:,1], '--', label='theta_d')
plt.xlabel('Time')
plt.ylabel('$\dot{x}, \dot{\\theta}$')
plt.legend()
plt.show()

