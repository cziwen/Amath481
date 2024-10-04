import numpy as np
import matplotlib.pyplot as plt
# Ziwen Chen


# ----------Q1----------


# Define the f(x) =  x*sin(3x) − exp(x)
def f(x):
    return x * np.sin(3 * x) - np.exp(x)


# Define the f'(x) = sin(3x) + 3x*cos(3x) - exp(x)
def df(x):
    return np.sin(3 * x) + 3 * x * np.cos(3 * x) - np.exp(x)

TOL = 1e-6


# Plot to see the function
dx = 0.05
x = np.arange(-1, 1+dx,dx)
plt.plot(x,f(x))
plt.axis([-1, 1, -1, 1])
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()


# Newton-Raphson method
x = np.array([-1.6])  # initial guess
for j in range(1000):
    x_new = x[j] - (f(x[j]) / df(x[j]))
    x = np.append(x, x_new)
    if abs(f(x[j])) < TOL:
        j += 1
        break

A1_iteration = j
A1 = x
print("A1:", A1)

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
        j += 1
        break

A2_iteration = j
A2 = c_n
print("A2:", c_n)

# 1 x 2 vector with the number of iterations for the Newton and bisection
A3 = [A1_iteration, A2_iteration]
print("A3:", A3)

# ----------Q2----------

# Define matrices and vectors
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1 ,0])
y = np.array([0, 1])
z = np.array([1, 2, -1])

# (a) A + B
A4 = A + B
print("\nA4 A + B = \n", A4)

# (b) 3x - 4y
A5 = 3 * x - 4 * y
print("\nA5 3x - 4y = \n", A5)

# (c) Ax
A6 = np.dot(A, x)
print("\nA6 Ax = \n", A6)

# (d) B(x - y)
A7 = np.dot(B, (x - y))
print("\nA7 B(x - y) = \n", A7)

# (e) Dx
A8 = np.dot(D, x)
print("\nA8 Dx = \n", A8)

# (f) Dy + z
A9 = np.dot(D, y) + z
print("\nA9 Dy + z = \n", A9)

# (g) AB
A10 = np.dot(A, B)
print("\nA10 AB = \n", A10)

# (h) BC
A11 = np.dot(B, C)
print("\nA11 BC = \n", A11)

# (i) CD
A12 = np.dot(C, D)
print("\nA12 CD = \n", A12)
