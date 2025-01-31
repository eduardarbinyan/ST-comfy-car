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

def create_road_input(s):
    u = np.zeros_like(time)
    start_bump1 = 5 / v
    end_bump1 = 9.5 / v
    start_bump2 = (5 + s) / v
    end_bump2 = (9.5 + s) / v

    bump1_indices = (time >= start_bump1) & (time <= end_bump1)
    bump2_indices = (time >= start_bump2) & (time <= end_bump2)

    z1 = v * time[bump1_indices]
    z2 = v * time[bump2_indices]

    u[bump1_indices] = 0.1 * np.sin((np.pi / (9.5 - 5)) * (z1 - 5))
    u[bump2_indices] = 0.1 * np.sin((np.pi / (9.5 - 5)) * (z2 - 5))
    
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

def analyze_response(s):
    road_input = create_road_input(s)
    response = simulate_response(road_input)
    
    y1_peaks = []
    y2_peaks = []
    diff_peaks = []
    
    for i in range(1, len(response[0])-1):
        if (response[0][i] > response[0][i-1] and response[0][i] > response[0][i+1]):
            y1_peaks.append((time[i], response[0][i]))
        if (response[1][i] > response[1][i-1] and response[1][i] > response[1][i+1]):
            y2_peaks.append((time[i], response[1][i]))
        if (response[2][i] > response[2][i-1] and response[2][i] > response[2][i+1]):
            diff_peaks.append((time[i], response[2][i]))
    
    print(f"\nAnalysis for s = {s} m:")
    print("Car Body Local Maxima:")
    for peak in y1_peaks:
        print(f"  Time: {peak[0]:.2f} s, Height: {peak[1]:.4f} m")
    
    print("Seat Local Maxima:")
    for peak in y2_peaks:
        print(f"  Time: {peak[0]:.2f} s, Height: {peak[1]:.4f} m")
    
    print("Seat-Body Difference Local Maxima:")
    for peak in diff_peaks:
        print(f"  Time: {peak[0]:.2f} s, Height: {peak[1]:.4f} m")
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, road_input, label='Road Elevation u(t)')
    plt.plot(time, response[0], label='Car Body Position y1(t)')
    plt.plot(time, response[1], label='Seat Position y2(t)')
    plt.plot(time, response[2], label='Difference y2(t) - y1(t)', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title(f'Response to Two Bumps with Gap s = {s} m')
    plt.legend()
    plt.grid()
    plt.show()
    
    return response

response_s15 = analyze_response(1.5)
response_s0 = analyze_response(0)