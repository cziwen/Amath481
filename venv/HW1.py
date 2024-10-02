from IPython.display import display
import numpy as np


# ----------Q1----------


# Define the f(x) =  x*sin(3x) − exp(x)
def f(x):
    return x * np.sin(3 * x) - np.exp(x)


# Define the f'(x) = sin(3x) + 3x*cos(3x) - exp(x)
def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)


TOL = 1e-6



# Newton-Raphson method
x = np.array([-1.6])  # initial guess
for j in range(1000):
    if abs(f(x[j])) < TOL:
        break
    x_new = x[j] - (f(x[j]) / df(x[j]))
    x = np.append(x, x_new)

print("A1:", x)

# Bisection method
a = -0.7
b = -0.4
c_n = np.array([])
for j in range(1000):
    c_new = (a + b) / 2

    f_n = f(c_new)

    if (f_n) > 0:
        a = c_new
    else:
        b = c_new

    c_n = np.append(c_n, c_new)

    if abs(f_n) < TOL:
        break

print("A2:", c_n)

# 1 x 2 vector with the number of iterations for the Newton and bisection
print("A3:", [len(x), len(c_n)])

# ----------Q2----------


# Define matrix
A = np.array([[1, 2],
              [-1, 1]])

B = np.array([[2, 0],
              [0, 2]])

C = np.array([[2, 0, -3],
              [0, 0, -1]])

D = np.array([[1, 2],
              [2, 3],
              [-1, 0]])

x = np.array([[1],
              [0]])

y = np.array([[0],
              [1]])

z = np.array([[1],
              [2],
              [-1]])

# print("\nMatrix A:")
# display(A)
#
# print("\nMatrix B:")
# display(B)
#
# print("\nMatrix C:")
# display(C)
#
# print("\nMatrix D:")
# display(D)
#
# print("\nx:")
# display(x)
#
# print("\ny:")
# display(y)
#
# print("\nz:")
# display(z)

# (a) A + B
print("\nA4: A + B = "), display(A + B)

# (b) 3x - 4y
print("\nA5: 3x - 4y = "), display(3 * x - 4 * y)

# (c) Ax
print("\nA6: Ax = "), display(np.dot(A, x))

# (d) B(x-y)
print("\nA7: B(x - y) = "), display(np.dot(B, (x - y)))

# (e) Dx
print("\nA8: Dx = "), display(np.dot(D, x))

# (f) Dy + z
print("\nA9: Dy + z = "), display(np.dot(D, y) + z)

# (g) AB
print("\nA10: AB = "), display(np.dot(A, B))

# (h) BC
print("\nA11: BC = "), display(np.dot(B, C))

# (i) CD
print("\nA12: CD = "), display(np.dot(C, D))
