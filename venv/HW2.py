# import numpy as np
# from scipy.integrate import odeint
# from scipy.integrate import RK45
# import matplotlib.pyplot as plt
# from IPython.display import display
#
#
# def shoot2(phi, x, n0, beta):
#     # This function returns [phi1', phi2']
#     # beta is the eigenValues
#     return [phi[1], (x**2 - beta) * phi[0]]
#
# tol = 1e-4  # define a tolerance level
# col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
# n0 = 100; phi0 = [0, 1]; L = 4
# xshoot =  np.arange(-L, L + 0.1, 0.1)  # x values for the integration with step size of 0.1
#
# eigenFunctions = []
# eigenValues = []
#
#
# beta_start = n0  # beginning value of beta
# for modes in range(1, 6):  # begin mode loop
#     beta = beta_start  # initial value of eigenvalue beta
#     dbeta = n0 / 100  # default step size in beta
#     for _ in range(1000):  # begin convergence loop for beta
#         y = odeint(shoot2, phi0, xshoot, args=(n0,beta))
#
#
#         if abs(y[-1, 0] - 0) < tol:  # check for convergence
#             eigenValues.append(beta)
#             eigenFunctions.append(y[:, 0])
#             break  # get out of convergence loop
#
#         if (-1) ** (modes + 1) * y[-1, 0] > 0:
#             beta -= dbeta
#         else:
#             beta += dbeta / 2
#             dbeta /= 2
#
#     beta_start = beta - 0.1  # after finding eigenvalue, pick new start
#     norm = np.trapezoid(y[:, 0] ** 2, xshoot)  # calculate the normalization
#     plt.plot(xshoot, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes
#
#
#
# plt.axis([-5, 5, -2, 2])
# plt.grid()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend(["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
# plt.show()  # end mode loop
#
#
# A1 = eigenFunctions
# A2 = eigenValues
#
# print("A1:")
# display(A1)
# print("A2:")
# display(A2)


# =================================== Potential Correct Version==================================
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt
from IPython.display import display


def shoot2 (phi, x, beta):
    # return phi_1' and phi_2'
    return phi[1], (x ** 2 - beta) * phi[0]


tol = 1e-6  # define a tolerance level
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
L = 4
xshoot = np.arange (-L, L + 0.1, 0.1)  # x values for the integration with step size of 0.1
K = 1  # Constant K given in the problem
beta_start = 0.1  # beginning value of beta

eigenFunctions = []
eigenValues = []



for modes in range (1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = 0.2  # default step size in beta

    for _ in range (1000):  # begin convergence loop for beta

        # Update initial conditions based on beta (ε_n)
        initialConditions = [1, np.sqrt (K * L ** 2 - beta)]

        # Integrate the ODE using the current initial conditions and beta
        y = odeint (shoot2, initialConditions, xshoot, args=(beta,))


        if abs(y[-1, 1] + np.sqrt (K * L ** 2 - beta) * y [-1, 0]) < tol:  # check for convergence at x = L
            break  # exit the convergence loop once eigenvalue is found

        # Adjust beta based on the boundary value at x = L
        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt (K * L ** 2 - beta) * y [-1, 0]) > 0:
            beta += dbeta
        else:
            beta -= dbeta
            dbeta /= 2


    beta_start = beta + 0.1  # adjust starting beta for next mode

    # Normalization
    norm = np.trapz (y[:, 0] ** 2, xshoot)  # calculate the normalization

    # save found values
    eigenValues.append (beta)
    eigenFunction = abs(y[:, 0] / np.sqrt (norm))
    eigenFunctions.append (eigenFunction)

    # plot
    plt.plot (xshoot, eigenFunction, col[modes - 1])  # plot modes



#plt.axis ([-5, 5, -1, 1])
plt.xlabel ("x")
plt.ylabel ("y")
plt.legend (["$\\phi_1$", "$\\phi_2$", "$\\phi_3$", "$\\phi_4$", "$\\phi_5$"], loc="upper right")
plt.grid ()
plt.show ()

A1 = np.column_stack(eigenFunctions)
A2 = eigenValues

print ("A1:", A1.shape)
display (A1)
print ("A2:")
display (A2)



