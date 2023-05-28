import numpy as np
import numpy.linalg as la

import matplotlib.pyplot as plt


# def rk4(f, x_k, u_k, h):
#     k_1 = f(x_k, u_k)


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

def record_system(simulation, u, total_time, time_step):
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
        u_t = u(simulation.x, time)
        simulation.update(u_t, time_step)
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

def calculate_K_with_poles(system, poles):
    """
    Calculates a K matrix for a system given poles. Only works for 2 state, 1 input
    systems (A is 2x2, B is 2x1).

    Derived by comparing the coefficients of the characteristic equation of A-BK and 
    the characterstic equation that result in the 2 given poles.

    returns the K matrix
    """
    if len(poles) != 2 or not (system.A.shape == (2,2) and system.B.shape == (2, 1)):
        raise ValueError("Pole placement not supported for systems that aren't 2 state, 1 input")

    A, B = system.A, system.B
    a, b, c, d = A[0][0], A[0][1], A[1][0], A[1][1]
    e, f = B[0][0], B[1][0]
    l1, l2 = poles

    m = -(l1 + l2) + (a + d)
    n = l1 * l2 - (a * d - b * c)

    x = b * f - e * d
    y = e * c - a * f

    # This system is derived from comparing the symbolic 
    # characteristic equations of A-BK and (lambda - l1)(lambda - l2)
    # 
    # [ e   f  | m ]
    # [ x   y  | n ]
    #   k1  k2 
    # 
    # It is row reduce to find k1, k2. 
    # 
    # The below code handles edge cases in row reduction, probably needs more
    # edge cases.

    if e != 0:
        k2 = (n * e - m * x)/(y * e - f * x)
        k1 = m/e - (f/e) * (k2)
    
        return np.array([ [k1, k2] ])
    else:
        k2 = m/f
        k1 = (n - k2 * y)/x

        return np.array([ [k1, k2] ])



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


    r = np.array([ 0, 0 ])
    poles = [ -2, -1 ]
    K = calculate_K_with_poles(system, poles)

    u = lambda x, t: K @ (r - x)

    total_time = 10
    time_step = 0.005

    plot_system(simulation, u, total_time, time_step)
