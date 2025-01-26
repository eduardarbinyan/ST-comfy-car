import numpy as np
import matplotlib.pyplot as plt

m1 = 1400
k1 = 1000
m2 = 100
k2 = 100
d2 = 200
d1_values = np.arange(250, 2001, 250)

time_end = 20
dt = 0.01
time = np.arange(0, time_end, dt)

u = np.zeros_like(time)
u[time >= 1] = 1

def state_derivative(x, d1, u_t):
    y1, y1_dot, y2, y2_dot = x
    y1_ddot = (1/m1) * (-k1 * y1 - d1 * y1_dot + d2 * (y2_dot - y1_dot) + k2 * (y2 - y1) + k1 * u_t)
    y2_ddot = (1/m2) * (-k2 * (y2 - y1) - d2 * (y2_dot - y1_dot))
    return np.array([y1_dot, y1_ddot, y2_dot, y2_ddot])

plt.figure(figsize=(10, 6))

for d1 in d1_values:
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y1_history = []
    for t in range(len(time)):
        y1_history.append(x[0])
        x_dot = state_derivative(x, d1, u[t])
        x += dt * x_dot
    plt.plot(time, y1_history, label=f'd1 = {d1} Ns/m')

plt.title('Car Body displacement (y1) for varying d1 Values')
plt.xlabel('Time (s)')
plt.ylabel('Displacement of Car Body y1 (m)')
plt.legend()
plt.grid()
plt.show()
