import numpy as np
import matplotlib.pyplot as plt

m1 = 1400
m2 = 100
d1 = 1000
d2 = 200
k1 = 1000
k2 = 100

y_star1 = 0.3
y_star2 = 0.6

def road_profile(z):
    if 5 <= z <= 9.5:
        return 0.1 * np.sin(np.pi * (z - 5) / (9.5 - 5))
    return 0

def euler_method(v, t_end):
    dt = 0.01
    steps = int(t_end / dt)
    
    x = np.zeros((4, steps))
    u = np.zeros(steps)
    
    t = np.linspace(0, t_end, steps)
    
    def dx_dt(x_curr, u_curr):
        x1, x2, dx1, dx2 = x_curr
        
        d_dx1 = (u_curr - k1*(x1 - y_star1) - d1*dx1 + k2*(x2 - x1 - (y_star2 - y_star1)) + d2*(dx2 - dx1)) / m1
        d_dx2 = (k2*(x1 - x2 + (y_star2 - y_star1)) + d2*(dx1 - dx2)) / m2
        
        return np.array([dx1, dx2, d_dx1, d_dx2])
    
    for i in range(1, steps):
        z_curr = v * t[i]
        u[i] = road_profile(z_curr)
        
        derivative = dx_dt(x[:, i-1], u[i-1])
        x[:, i] = x[:, i-1] + dt * derivative
    
    return t, x, u

v1 = 3
v2 = 10

t_end = 20

t1, x1, u1 = euler_method(v1, t_end)
t2, x2, u2 = euler_method(v2, t_end)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title(f'Car response for v = {v1} m/s')
plt.plot(t1, u1 + y_star1, label='Road Surface')
plt.plot(t1, x1[0,:] + y_star1, label='Car Body')
plt.plot(t1, x1[1,:] + y_star2, label='Chair')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

plt.subplot(2, 1, 2)
plt.title(f'Car response for v = {v2} m/s')
plt.plot(t2, u2 + y_star1, label='Road Surface')
plt.plot(t2, x2[0,:] + y_star1, label='Car Body')
plt.plot(t2, x2[1,:] + y_star2, label='Chair')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()

plt.tight_layout()
plt.show()

seat_body_diff_slow = np.max(x1[1,:] - x1[0,:])
seat_body_diff_fast = np.max(x2[1,:] - x2[0,:])

print(f"Maximum seat elevation (slow speed): {seat_body_diff_slow:.4f} m")
print(f"Maximum seat elevation (fast speed): {seat_body_diff_fast:.4f} m")