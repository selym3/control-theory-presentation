import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

class System:
    def __init__(self, A, x):
        self.A = A
        self.x = x

    def update(self, dt):
        dx_dt = self.A @ self.x
        self.x = self.x + ((dx_dt) * dt)

        return self.x, dx_dt

def plot_system(system, iters, dt): # integrates really poorly
    time = 0

    X = []
    Y = []
    U = []
    V = []

    for _ in range(iters):
        # print(f'({spring.x[0]}, {spring.x[1]}), ({time}, {spring.x[0]}), ({time}, {spring.x[1]}), ', end='')
        x, dx_dt = system.update(dt)

        X.append(x[0])
        Y.append(x[1])
        U.append(dx_dt[0])
        V.append(dx_dt[1])


        time += dt

    plt.quiver(X, Y, U, V)
    plt.show()

def plot_system_2(system, iterations, dt):
    eigvals, eigvecs = la.eig(system.A)
    def x(t):
        return sum(np.exp(value * t) * vector for value, vector in zip(eigvals, eigvecs))
    
    time = 0
    X = []
    Y = []

    for i in range(iterations):
        state = np.real(x(time))

        X.append(state[0])
        Y.append(state[1])

        time += dt

    plt.plot(X, Y)
    plt.show()


def plot_system_3(system, total_time, dt):
    eigvals, eigvecs = la.eig(system.A)
    def x(t):
        return sum(np.exp(value * t) * vector for value, vector in zip(eigvals, eigvecs))
    
    T=[]
    X=[]

    time = 0
    while time <= total_time:
        state = np.real(x(time))[0]
        T.append(time)
        X.append(state)
        time += dt

    return T, X

k = 1.2
b = 0.0
m = 1.0

A = np.array([[ 0, 1 ],
              [-k/m, -b/m]])

x = np.array([ 1, 0 ])


spring = System(A, x)

dt = 0.1
time = 0

fig, ax = plt.subplots()

T, X = plot_system_3(spring, 10, dt)
text = f'k={k:.3}N/m, m={m:.3}kg, b={b:0.3}Ns/m'
ax.plot(T, X)
ax.set_title('Spring position vs time (' + (r'$' + text +'$)'))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
plt.show()

# import time
# fig.savefig(f'./output-{int(time.time() * 10**4)}.png', dpi=75, bbox_inches='tight', pad_inches=0.1)
