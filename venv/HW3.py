import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_bvp, simpson, solve_ivp
from scipy.sparse.linalg import eigs


# ============================  (A) =============================

def shoot2 (x, phi, beta):
    # Return phi_1' and phi_2'
    return [phi[1], (x ** 2 - beta) * phi[0]]


tol = 1e-6  # Define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # Eigenfunction colors
L = 4
xshoot = np.arange (-L, L + 0.1, 0.1)  # x values for integration with step size of 0.1
K = 1  # Constant K given in the problem
beta_start = 0.1  # Beginning value of beta

eigenFunctions = []
eigenValues = []

for modes in range (1, 6):  # Begin mode loop
    beta = beta_start  # Initial value of eigenvalue beta
    dbeta = 0.2  # Default step size in beta

    for _ in range (1000):  # Begin convergence loop for beta

        # Update initial conditions based on beta (ε_n)
        initialConditions = [1, np.sqrt (K * L ** 2 - beta)]

        # Integrate the ODE using solve_ivp with the current initial conditions and beta
        sol = solve_ivp (shoot2, [-L, L + 0.1], initialConditions, args=(beta,), t_eval=xshoot)

        if abs (sol.y[1, -1] + np.sqrt (K * L ** 2 - beta) * sol.y[0, -1]) < tol:  # Check for convergence at x = L
            break  # Exit the convergence loop once eigenvalue is found

        # Adjust beta based on the boundary value at x = L
        if (-1) ** (modes + 1) * (sol.y[1, -1] + np.sqrt (K * L ** 2 - beta) * sol.y[0, -1]) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2

    beta_start = beta + 0.1  # Adjust starting beta for the next mode

    # Normalization
    norm = np.trapz (sol.y[0] ** 2, sol.t)  # Calculate the normalization

    # Save found values
    eigenValues.append (beta)
    eigenFunction = abs (sol.y[0] / np.sqrt (norm))
    eigenFunctions.append (eigenFunction)

    # Plot
    plt.plot (sol.t, eigenFunction, col[modes - 1])  # Plot modes

plt.xlabel ("x")
plt.ylabel ("y")
plt.legend (["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
plt.grid ()
plt.show ()

A1 = np.column_stack (eigenFunctions)
A2 = eigenValues

print ("A1", A1)
print ("A2", A2)

# ========================== (B) ==========================

# HW 3 - Part b
L = 4
dx = 0.1
x = np.arange (-L, L + dx, dx)
N = len (x) - 2

A = np.zeros ((N, N))
for j in range (N):
    A[j, j] = -2 - (x[j + 1] ** 2) * (dx ** 2)

for j in range (N - 1):
    A[j, j + 1] = 1
    A[j + 1, j] = 1

A_1 = A

A_2 = np.zeros ((N, N))
A_2[0, 0] = 4 / 3
A_2[0, 1] = - 1 / 3

A_3 = np.zeros ((N, N))
A_3[N - 1, N - 2] = - 1 / 3
A_3[N - 1, N - 1] = 4 / 3

A = A_1 + A_2 + A_3
A = A / (dx ** 2)

# Compute eigenvalues and eigenvectors
D, V = eigs (- A, k=5, which='SM')

# calculate the bc's
phi_0 = 4 / 3 * V[0, :] - 1 / 3 * V[1, :]
phi_n = - 1 / 3 * V[-2, :] + 4 / 3 * V[-1, :]

# append to the side
V = np.vstack ((phi_0, V, phi_n))

# normalize and plot
for i in range (5):
    norm = np.trapz (V[:, i] ** 2, x)  # calculate the normalization
    V[:, i] = abs (V[:, i] / np.sqrt (norm))

    plt.plot (x, V[:, i])

plt.legend (["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
plt.show ()

# assign to A3 and A4
A3 = V
A4 = D

print ("A3", A3)
print ("A4", A4)


# =================================(c)=====================================

# HW 3 - Part c

# Define differential equation
def shoot_eq (x, phi, epsilon, gamma):
    # return phi', phi''
    return [phi[1],
            (gamma * phi[0] ** 2 + x ** 2 - epsilon) * phi[0]]


# Parameters
tol = 1e-6
L = 2
dx = 0.1
xshoot = np.arange (-L, L + dx, dx)  # range of x values
gamma_values = [0.05, - 0.05]

# Setting matrix
A5, A7 = np.zeros ((len (xshoot), 2)), np.zeros ((len (xshoot), 2))
A6, A8 = np.zeros (2), np.zeros (2)

# Gamma loops
for gamma in gamma_values:
    epsilon_start = 0.1
    A = 1e-6

    # main loop
    for modes in range (1, 3):
        dA = 0.01

        # Iterations to adjust A
        for _ in range (100):
            epsilon = epsilon_start
            depsilon = 0.2

            # Iterations to adjust epsilon
            for i in range (100):
                # initial conditions
                phi0 = [A, np.sqrt (L ** 2 - epsilon) * A]

                # Solve the ODE
                ans = solve_ivp (
                    lambda x, phi: shoot_eq (x, phi, epsilon, gamma),
                    [xshoot[0], xshoot[-1]],
                    phi0,
                    t_eval=xshoot
                )
                phi_sol = ans.y.T
                x_sol = ans.t

                # Check boundary condition
                bc = phi_sol[-1, 1] + np.sqrt (L ** 2 - epsilon) * phi_sol[-1, 0]
                if abs (bc) < tol:
                    break

                # Adjust to steps of epsilon
                if (-1) ** (modes + 1) * bc > 0:
                    epsilon += depsilon
                else:
                    epsilon -= depsilon
                    depsilon /= 2

            # Check whether it is focused
            integral = simpson (phi_sol[:, 0] ** 2, x=x_sol)
            if abs (integral - 1) < tol:
                break

            # Adjust to steps of A
            if integral < 1:
                A += dA
            else:
                A -= dA
                dA /= 2

        # Adjust to epsilon_start
        epsilon_start = epsilon + 0.2

        # Input results of eigenfuncitons & eigenvalues
        if gamma > 0:
            A5[:, modes - 1] = np.abs (phi_sol[:, 0])
            A6[modes - 1] = epsilon

        else:
            A7[:, modes - 1] = np.abs (phi_sol[:, 0])
            A8[modes - 1] = epsilon

plt.plot (xshoot, A5)
plt.plot (xshoot, A7)
plt.legend (["$\\phi_1$", "$\\phi_2$"], loc="upper right")
print ("A5", A5)
print ("A7", A7)

print ("A6:")
print (A6)
print ("A8:")
print (A8)
plt.show ()


# =========================== (D) ==========================
def hwl_rhs_a (x, y, beta):
    return [y[1], (x ** 2 - beta) * y[0]]


L = 2
x_span = [-L, L]
beta = 1
A = 1
y0 = [A, np.sqrt (L ** 2 - beta) * A]
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

dt45, dt23, dtRadav, dtBDF = [], [], [], []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}

    sol45 = solve_ivp (hwl_rhs_a, x_span, y0, method='RK45', args=(beta,), **options)
    sol23 = solve_ivp (hwl_rhs_a, x_span, y0, method='RK23', args=(beta,), **options)
    solRadav = solve_ivp (hwl_rhs_a, x_span, y0, method='Radau', args=(beta,), **options)
    solBDF = solve_ivp (hwl_rhs_a, x_span, y0, method='BDF', args=(beta,), **options)

    # calculate avg time steps, for each method
    dt45.append (np.mean (np.diff (sol45.t)))
    dt23.append (np.mean (np.diff (sol23.t)))
    dtRadav.append (np.mean (np.diff (solRadav.t)))
    dtBDF.append (np.mean (np.diff (solBDF.t)))

# perform linear regression (log - log) to determine slopes
fit45 = np.polyfit (np.log (dt45), np.log (tols), 1)
fit23 = np.polyfit (np.log (dt23), np.log (tols), 1)
fitRadav = np.polyfit (np.log (dtRadav), np.log (tols), 1)
fitBDF = np.polyfit (np.log (dtBDF), np.log (tols), 1)

# extract slopes
slope45 = fit45[0]
slope23 = fit23[0]
slopeRadav = fitRadav[0]
slopeBDF = fitBDF[0]

A9 = np.array ([slope45, slope23, slopeRadav, slopeBDF])

print ("A9:", A9)


# ========================== E ==========================

# HW 3 - Part e
# Define first five Gauss-Hermite polynomial
def H0 (x):
    return np.ones_like (x)


def H1 (x):
    return 2 * x


def H2 (x):
    return 4 * (x ** 2) - 2


def H3 (x):
    return 8 * (x ** 3) - 12 * x


def H4 (x):
    return 16 * (x ** 4) - 48 * (x ** 2) + 12


def factorial (n):
    result = 1
    for i in range (1, n + 1):
        result *= i
    return result


# Define x range
L = 4
dx = 0.1
x = np.arange (-L, L + dx, dx)

# Create matrix h
h = np.column_stack ([H0 (x), H1 (x), H2 (x), H3 (x), H4 (x)])

phi = np.zeros (h.shape)

# solve for phi(exact)
for j in range (5):
    phi[:, j] = ((np.exp (- (x ** 2) / 2) * h[:, j]) /
                 np.sqrt (factorial (j) * (2 ** j) * np.sqrt (np.pi))
                 )

erps_a = np.zeros (5)
erps_b = np.zeros (5)

er_a = np.zeros (5)
er_b = np.zeros (5)

for j in range (5):  # Compute errors
    # compute eigen func
    erps_a[j] = np.trapz (((abs (A1[:, j])) - (abs (phi[:, j]))) ** 2, x=x)
    erps_b[j] = np.trapz (((abs (A3[:, j])) - (abs (phi[:, j]))) ** 2, x=x)

    # compute eigen values
    er_a[j] = 100 * (abs (A2[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))
    er_b[j] = 100 * (abs (A4[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))

# for j in range (5):
#     erps_a[j] = simpson (((abs (A1[:, j])) - abs (phi[:, j])) ** 2, x=x)
#     erps_b[j] = simpson (((abs (A3[:, j])) - abs (phi[:, j])) ** 2, x=x)
#
#     er_a[j] = 100 * (abs (A2[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))
#     er_b[j] = 100 * (abs (A4[j] - (2 * (j + 1) - 1)) / (2 * (j + 1) - 1))

A10 = erps_a
A11 = er_a

A12 = erps_b
A13 = er_b

print ("A10: ", A10)
print ("A11: ", A11)

print ("A12:", A12)
print ("A13", A13)
