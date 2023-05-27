import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt

class LinearSystem:
    """
    Defines a linear system, x' = Ax + Bu, in terms of 
    its matrices A and B
    
    A - state dynamics matrix
    B - input dynamics matrix
    """
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def calculate(self, x, u):
        return self.A @ x + self.B @ u

class LinearSystemSim:
    """
    Uses a linear system to simulate a system over time

    system - linear system to use
    x_0 - initial conditions
    """
    def __init__(self, system, x_0):
        self.system = system
        self.x = x_0

    def update(self, u_t, dt):
        # numerical integration
        dx_dt = self.system.calculate(self.x, u_t)
        self.x = self.x + dx_dt * dt

# def rk4(f, x_k, u_k, h):
#     k_1 = f(x_k, u_k)

def record_system(simulation, u_t, total_time, time_step):
    """
    Records system inputs and states over time

    simulation - linear system simulation
    u_t - function that takes in time and returns an input vector
    total_time - total time to run simulation for
    time_step - time increment (dt)

    return (time array), (input array [first element]), (state array [first element])
    """
    T = [0]
    U = [0]
    X = [simulation.x[0]]
    
    time = 0
    while time <= total_time:
        u = u_t(time)
        simulation.update(u, time_step)
        time += time_step

        T.append(time)
        U.append(u)
        X.append(simulation.x[0])

    return T, U, X

def plot_system(simulation, u_t, total_time, time_step):
    """
    Opens a matplotlib plot to plot the system over time
    """
    fig, ax = plt.subplots()

    T, U, X = record_system(simulation, u_t, total_time, time_step)
    ax.plot(T, X)

    fig.show()
    plt.show()

if __name__ == "__main__":
    
    m = 1.0
    k = 1.2
    b = 0.0

    A = np.array([ [ 0, 1 ] ,
                 [ -k/m, -b/m ]])
    
    B = np.array([ [ 0 ],
                   [ 1/m ] ])
    
    system = LinearSystem(A, B)

    x_0 = np.array([ 1, 0 ])

    simulation = LinearSystemSim(system, x_0)

    u_t = lambda t: np.array([ 1.2 * np.sin(t) ])

    total_time = 10
    time_step = 0.005

    plot_system(simulation, u_t, total_time, time_step)
