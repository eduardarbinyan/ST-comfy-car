import numpy as np
import matplotlib.pyplot as plt

m1, d1, k1 = 1400, 1000, 1000
m2, d2, k2 = 100, 50, 50

A = np.array([
    [0, 1, 0, 0],
    [-k1/m1 - k2/m1, -d1/m1 - d2/m1, k2/m1, d2/m1],
    [0, 0, 0, 1],
    [k2/m2, d2/m2, -k2/m2, -d2/m2]
])
B = np.zeros((4, 1)) 
C = np.array([
    [1, 0, 0, 0],  
    [0, 0, 1, 0]   
])

initial_conditions = [
    [0, -0.2, 0, 0],  
    [0, 0, 0, -0.2]   
]

dt = 0.01
t_max = 20
time_steps = int(t_max / dt)
t = np.linspace(0, t_max, time_steps)

for x0 in initial_conditions:
    x = np.zeros((4, time_steps))
    x[:, 0] = x0
    for i in range(1, time_steps):
        x[:, i] = x[:, i-1] + dt * A @ x[:, i-1]
    
    y = C @ x
    y1 = 0.3 + y[0, :]  
    y2 = 0.6 + y[1, :]  
    
    plt.plot(t, y1, label='y1(t)')
    plt.plot(t, y2, label='y2(t)')
    plt.title(f"Initial state: {x0}")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid()
    plt.show()
