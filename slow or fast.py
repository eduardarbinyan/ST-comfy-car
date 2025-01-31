import numpy as np
import matplotlib.pyplot as plt

m1 = 1400
k1 = 1000
m2 = 100
k2 = 100
d1 = 1000
d2 = 200
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

def simulate_response(road_input):
    x = np.array([0.0, 0.0, 0.0, 0.0])
    y1_history = []
    y2_history = []
    difference_history = []
    for t in range(len(time)):
        y1_history.append(x[0])
        y2_history.append(x[2])
        difference_history.append(x[2] - x[0])
        x_dot = state_derivative(x, road_input[t])
        x += dt * x_dot
    return y1_history, y2_history, difference_history

def analyze_response(v):
    road_input = create_road_input(v)
    response = simulate_response(road_input)
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, road_input, label='Road Elevation u(t)')
    plt.plot(time, response[0], label='Car Body Position y1(t)')
    plt.plot(time, response[1], label='Seat Position y2(t)')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'Response to Speed Bump at v = {v} m/s')
    plt.legend()
    plt.grid()
    plt.show()
    
    max_seat_elevation = np.max(np.abs(np.array(response[1]) - np.array(response[0])))
    print(f'Maximum seat elevation for v = {v} m/s: {max_seat_elevation:.4f} m')
    
    return response

print("Simulation for v = 3 m/s:")
response_slow = analyze_response(3)

print("\nSimulation for v = 10 m/s:")
response_fast = analyze_response(10)