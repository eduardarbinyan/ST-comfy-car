import numpy as np
import matplotlib.pyplot as plt

m1 = 1400
k1 = 1000
m2 = 100
k2 = 100
d1 = 1000
d2 = 200
v = 3
time_end = 20
dt = 0.01
time = np.arange(0, time_end, dt)

u = np.zeros_like(time)
start_bump = 5 / v
end_bump = 9.5 / v
bump_indices = (time >= start_bump) & (time <= end_bump)
z = v * time[bump_indices]
u[bump_indices] = 0.1 * np.sin((np.pi / 5) * (z - 5))

def state_derivative(x, u_t):
    y1, y1_dot, y2, y2_dot = x
    y1_ddot = (1/m1) * (-k1 * y1 - d1 * y1_dot + d2 * (y2_dot - y1_dot) + k2 * (y2 - y1) + k1 * u_t)
    y2_ddot = (1/m2) * (-k2 * (y2 - y1) - d2 * (y2_dot - y1_dot))
    return np.array([y1_dot, y1_ddot, y2_dot, y2_ddot])

x = np.array([0.0, 0.0, 0.0, 0.0])
y1_history = []
y2_history = []
difference_history = []

for t in range(len(time)):
    y1_history.append(x[0])
    y2_history.append(x[2])
    difference_history.append(x[2] - x[0])
    x_dot = state_derivative(x, u[t])
    x += dt * x_dot

plt.figure(figsize=(10, 6))
plt.plot(time, u, label='Road Elevation u(t)')
plt.plot(time, y1_history, label='Car Body Position y1(t)')
plt.plot(time, y2_history, label='Seat Position y2(t)')
plt.plot(time, difference_history, label='Difference y2(t) - y1(t)', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Response of Car and Seat to a Single Speed Bump')
plt.legend()
plt.grid()
plt.show()

max_elevation_difference = np.max(np.array(y2_history) - np.array(y1_history))
print(f'Maximum elevation difference between seat and car body: {max_elevation_difference:.4f} m')
    