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
g = 9.81

def calculate_dydt(y,t,M,m,L):

    x_d = y [1] 
    theta = y[2]
    theta_d = y[3]
    
    
    M_matrix= np.array( [ [M+m*L, -0.5*m*L**2*np.sin(theta)],

                          [ -0.5*m*L**2*np.sin(theta), m*L**3/3]
                          ])
    
    F = np.array([[0.5*m*L**2*theta_d**2*np.cos(theta)],
                  [0.5*m*L**2*g*np.cos(theta)]
                  ])
    
    acc = np.linalg.inv (M_matrix) @ F
    
    x_dd, theta_dd = acc.flatten()

    dy_dt = [x_d, x_dd, theta_d, theta_dd]

    return dy_dt

sol = odeint (calculate_dydt,Y0,t, args=(M,m,L))

plt.figure()
plt.plot(t,sol[:,0],label = 'x')
plt.plot(t,sol[:,2],label = 'theta')
plt.legend()

plt.figure()
plt.plot(t,sol[:,1],label = 'x_d')
plt.plot(t,sol[:,3],label = 'theta_d')
plt.legend()
plt.show()


t = 0
dt = 0.1
y = np.zeros(2)
y_d = np.zeros(2)


def calculate_g(y,t_d,M,m,L):

    x = y[0]
    x_d = y_d[0]  

    theta = y[1]
    theta_d = y_d[1]
    
    M_matrix= np.array( [ [M+m*L, -0.5*m*L**2*np.sin(theta)],

                          [ -0.5*m*L**2*np.sin(theta), m*L**3/3]
                          ])
    
    F = np.array([[0.5*m*L**2*theta_d**2*np.cos(theta)],
                  [0.5*m*L**2*g*np.cos(theta)]
                  ])
    
    g = np.linalg.inv (M_matrix) @ F

    return g

def rungeKutta(dt,y,y_d,M,m,L):

    y_dd = calculate_g(y,y_d,M,m,L)

    A = dt*y_dd/2
    b = dt *(y_d+0.5*A)/2
    


     


    return
