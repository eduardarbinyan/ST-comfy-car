import numpy as np
import control

m1 = 1400
d1 = 1000 
k1 = 1000
m2 = 100
d2 = 50
k2 = 50

A = np.array([[ 0,  0,  1,  0],
              [ 0,  0,  0,  1],
              [-k1/m1, k2/m1, -d1/m1, d2/m1],
              [ k2/m2,-k2/m2, d2/m2,-d2/m2]])

B = np.array([[ 0],
              [ 0],
              [ 1/m1],
              [-1/m2]])

C = np.array([[ 0, 1, 0, 0],
              [ 0, 0, 0, 1]])

Wc = control.ctrb(A, B)
print(f"Rank of controllability matrix: {np.linalg.matrix_rank(Wc)}")

x0 = np.array([0, -0.2, 0, 0])
t = np.linspace(0, 20, 1000)
t, y = control.forced_response(control.ss(A, B, C, 0), t, np.zeros_like(t), x0)

y1_uncontrolled = y[:, 0]
y2_uncontrolled = y[:, 1]

Q = np.diag([1, 1, 1, 1])
R = 1
K, _, _ = control.lqr(A, B, Q, R)
F = K

t, y = control.forced_response(control.ss(A - B@F, B, C, 0), t, np.zeros_like(t), x0)
y1_controlled = y[:, 0]
y2_controlled = y[:, 1]
