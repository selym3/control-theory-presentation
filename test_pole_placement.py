# tests my solution for 2 state, 1 input pole placement

import numpy as np


A = np.array([ [ 0, 1 ],
               [ -1.2, 0]])

B = np.array([ [ 0 ], 
               [ 1 ]])

a = A[0][0]
b = A[0][1]
c = A[1][0]
d = A[1][1]


# print(a, b, c, d)

e = B[0][0]
f = B[1][0]

# print(e, f)

l1 = -2
l2 = -1

m = -(l1 + l2) + (a + d)
n = l1 * l2 - (a * d - b * c)

x = b * f - e * d
y = e * c - a * f

print(y)

# [ e   f  | m ]
# [ x   y  | n ]
#   k1  k2 
# row reduce to find k1, k2

if e != 0:
    k2 = (n * e - m * x)/(y * e - f * x)
    k1 = m/e - (f/e) * (k2)
  
    print(k1, k2)
else:
    k2 = m/f
    print(m, f)
    k1 = (n - k2 * y)/x
    print(k1, k2)