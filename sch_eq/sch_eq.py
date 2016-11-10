import sys
import os
#make sure the program can be executable from test file
dir_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(dir_root)


import numpy as np
import matplotlib.pyplot as plt
import math as mt
import numpy.polynomial.legendre as legen
import cmath
from function import wave_fourier_basis, Hamiltonian_momentum_basis, Legendre_polynomial_basis, reconstruct_wave

#input includes:
#c the constant
#N the size of the basis set
#V the potential energy V(x) ps: the size of V(x) should be same as the size of the basis set
#V_const the constant potential energy 
#domain is the range of V(x)
#the choice of basis set function: 1 ---> the fourier basis                                                            2 ---> the legendre polynomial basis
#ps: the fourier basis must take V as a function of x, which means the V input must be a array, and the legendre polynomial basis  can only take the constant V. Be careful when you use different basis method
from scipy.linalg import dft


def output(c, V, V_const, N, wave_func, choice, domain):
    if choice == 1:
        matrix = Hamiltonian_momentum_basis(c, V, domain, N)
        wave_fourier = wave_fourier_basis(wave_func, domain, N)
        result = np.dot(matrix, wave_fourier)
        return result

    elif choice == 2:
        return Legendre_polynomial_basis(c, V_const, domain, N, wave_func) 
    else:
        return "error, only two basis are available, please select from 1 or 2"


#rough test
##set V = 0, and input wavefunction is a sin function, using the column              vector we obtained in legendre polynomial basis and fourier basis, we could         restruct the sin function
#set up
N = 100
domain = 2 * np.pi
x = np.linspace(-domain / 2,domain / 2,N)

wave_func = np.sin(x)
potential = np.repeat(0, N)

#fourier basis
c1 = output(1, potential, 0, N, wave_func, 1, domain)
a1 = reconstruct_wave(c1, domain, N)
a1 = a1.real

#Legen_polynomial
c2 = output(1, potential, 0, N, wave_func, 2, domain)
a2 = legen.legval(x, c2)

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(x, wave_func, x, a1)

plt.figure(2)
plt.plot(x, wave_func, x, a2)
plt.show()
