import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

def record_system(system, x, total_time, dt):
    T=[0]
    X=[x[0]]

    time = 0
    while time <= total_time:
        time += dt
        T.append(time)

        dx_dt = system @ x
        x1 = x[1] + dx_dt[1] * dt
        x0 = x[0] + x1 * dt
        x=np.array([ x0, x1 ])

        X.append(x[0])
        

    return T, X

def record_system_2(A, B, K, r, x, total_time, dt):
    T=[0]
    X=[x[0]]

    time = 0
    while time <= total_time:
        time += dt
        T.append(time)

        # solve for control input
        u = K @ (r - x)

        # calculate change in state 
        dx_dt = A @ x + B @ u

        # numerically integrate (in a good way)
        x1 = x[1] + dx_dt[1] * dt
        x0 = x[0] + x1 * dt

        x=np.array([ x0, x1 ])

        X.append(x[0])
        

    return T, X

def record_system_3(A, B, F, r, x, total_time, dt):
    T=[0]
    X=[x[0]]

    time = 0
    while time <= total_time:
        time += dt
        T.append(time)

        # solve for control input
        u = F if (r[0] - x[0]) > 0 else -F

        # calculate change in state 
        dx_dt = A @ x + B @ u

        # numerically integrate (in a good way)
        x1 = x[1] + dx_dt[1] * dt
        x0 = x[0] + x1 * dt

        x=np.array([ x0, x1 ])

        X.append(x[0])
        

    return T, X


k = 1.2
b = 0.0
m = 1.0

A = np.array([[ 0, 1 ],
              [-k/m, -b/m]])

B = np.array([ [ 0 ], 
               [1.0/m ]])

K = np.array([ [3, 0.6] ])# [ 120 - 1.2, 22 ] ])
                # [0.8, 3] ])

x = np.array([ 1, 0 ])

r = np.array([ 0, 0 ])


print(la.eig(A - B @ K))

dt = 0.01
time = 0

fig, ax = plt.subplots()

F = np.array([ 5 ])

T, X = record_system_2(A, B, K, r, x, 10, dt)
# T, X = record_system(A, x, 10, dt)
text = f'k={k:.3}N/m, m={m:.3}kg, b={b:0.3}Ns/m'
ax.plot(T, X)
ax.set_title('Spring position vs time (' + (r'$' + text +'$)'))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
plt.show()

import sys
if len(sys.argv) == 2:
    fig.savefig(f'./images/{sys.argv[1]}', dpi=75, bbox_inches='tight', pad_inches=0.1)
