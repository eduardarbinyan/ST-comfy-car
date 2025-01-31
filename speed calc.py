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

def create_road_input(v):
    u = np.zeros_like(time)
    start_bump = 5 / v
    end_bump = 9.5 / v

    bump_indices = (time >= start_bump) & (time <= end_bump)
    z = v * time[bump_indices]

    u[bump_indices] = 0.1 * np.sin((np.pi / (9.5 - 5)) * (z - 5))
    
    return u

def state_derivative(x, u_t):
    y1, y1_dot, y2, y2_dot = x
    y1_ddot = (1/m1) * (-k1 * y1 - d1 * y1_dot + d2 * (y2_dot - y1_dot) + k2 * (y2 - y1) + k1 * u_t)
    y2_ddot = (1/m2) * (-k2 * (y2 - y1) - d2 * (y2_dot - y1_dot))
    return np.array([y1_dot, y1_ddot, y2_dot, y2_ddot])

def simulate_response():
    road_input = create_road_input(v)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y1_history = [0.0]
    y2_history = [0.0]
    y1_dot_history = [0.0]
    y2_dot_history = [0.0]

    for t in range(1, len(time)):
        x_dot = state_derivative(x, road_input[t-1])
        x += dt * x_dot
        
        y1_history.append(x[0])
        y2_history.append(x[2])
        y1_dot_history.append(x[1])
        y2_dot_history.append(x[3])

    return road_input, y1_history, y2_history, y1_dot_history, y2_dot_history

road_input, y1, y2, y1_dot, y2_dot = simulate_response()

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.title('Road Elevation and Positions')
plt.plot(time, road_input, label='Road Elevation u(t)')
plt.plot(time, y1, label='Car Body Position y1(t)')
plt.plot(time, y2, label='Seat Position y2(t)')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.title('Speeds of Car Body and Seat')
plt.plot(time, y1_dot, label='Car Body Speed ẏ1(t)')
plt.plot(time, y2_dot, label='Seat Speed ẏ2(t)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

def check_differentiability(speeds):
    accelerations = np.diff(speeds) / dt
    max_acceleration = np.max(np.abs(accelerations))
    print(f"Maximum acceleration: {max_acceleration:.4f} m/s²")
    print("Technically non-differentiable (discrete jumps in acceleration)")

print("\nCar Body Speed Differentiability:")
check_differentiability(y1_dot)

print("\nSeat Speed Differentiability:")
check_differentiability(y2_dot)