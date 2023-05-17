import numpy as np
import numpy.linalg as la

k = 3
b = 0.1

A = np.matrix([ [0, 1] , [-k, -b] ])
# print(A)

# eigvals, eigvecs = la.eig(A)


# print(eigvals)
# print(eigvecs)

def find_solution(A):
    eigvals, eigvecs = la.eig(A)
    def x(t):
        return sum(np.exp(value * t) * vector for value, vector in zip(eigvals, eigvecs))
    return x

# print("\nSolutions: \n")
x = find_solution(A)

t = 0
while t <= 5:
    state = np.real(x(t))
    x_ = state[0,0]
    v_ = state[0,1]

    print(f'({t}, {x_})')
    # print(f'({t}, {v_})')
    t += 0.01

# print(np.real(x(0.5)[0,0]))