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
from function import wave_fourier_basis, Hamiltonian_momentum_basis, Legendre_polynomial_basis, reconstruct_wave, \
                    Hamiltonian_Legendre_polynomial

#input includes:
#c the constant
#N the size of the basis set
#V the potential energy V(x) ps: the size of V(x) should be same as the size of the basis set
#V_const the constant potential energy 
#domain is the range of V(x)
#the choice of basis set function: 1 ---> the fourier basis     2 ---> the legendre polynomial basis
#ps: the fourier basis must take V as a function of x, which means the V input must be a array, and the legendre polynomial basis  can only take the constant V. Be careful when you use different basis method

def output(c, V, V_const, N, wave_func, choice, domain):
    if choice == 1:
        matrix = Hamiltonian_momentum_basis(c, V, domain, N)
        wave_fourier = wave_fourier_basis(wave_func, domain, N)
        result = np.dot(matrix, wave_fourier)
        return result

    elif choice == 2:
        wave_legen =  Legendre_polynomial_basis(c, domain, N, wave_func)
        matrix = Hamiltonian_Legendre_polynomial(c, V, domain, N)
        wave_result_legen = np.dot(matrix, wave_legen)
        return wave_result_legen
    else:
        return "error, only two basis are available, please select from 1 or 2"
    
def ground_wave_function(c, V, domain, N):
    matrix = Hamiltonian_momentum_basis(c, V, domain, N)
    w, v = np.linalg.eig(matrix)
    w = w.real
    idx = w.argsort()
    w = w[idx]
    v = v[:,idx]
    ground_wave = reconstruct_wave(v[:,0], domain, N)
    return ground_wave
#rough test
##set V = 0, and input wavefunction is a sin function, using the column vector we obtained in legendre polynomial basis and fourier basis, we could reconstruct the sin function
#set up
N = 50
domain = 2
x = np.linspace(-domain / 2,domain / 2, N)

#wave_func = np.sin(x)
#potential = np.repeat(0, N)
#
##fourier basis
#c1 = output(1, potential, 0, N, wave_func, 1, domain)
#a1 = reconstruct_wave(c1, domain, N)
#a1 = a1.real
#
##Legen_polynomial
#c2 = output(1, potential, 0, N, wave_func, 2, domain)
#a2 = legen.legval(x, c2)
#
#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.plot(x, wave_func, x, a1)
#
#
#plt.figure(2)
#plt.plot(x, a2)

#potential = x ** 2 / 2
potential = 5

h = Hamiltonian_Legendre_polynomial(1, potential, domain, N)
f, g = np.linalg.eig(h)
idx = f.argsort()
f = f[idx]
g = g[:,idx]
print(f)
print(h)
##plotting out the gournd state funciton using fourier basis for harmonic oscillator
#plt.figure(3)
#w = ground_wave_function(1 / 2, potential, domain, N)
#w = w.real
#plt.plot(x, w)
#plt.show()

#m = 1
#omega = 1
#c = 1 / 2 / m 
#domain = 15
#N = 70
#
##the corresponding potential
#x  = np.linspace(-1 * domain / 2, domain / 2, N)
#potential = m * omega ** 2 * x ** 2 / 2
#
##from analytic solution, the first five energy are
#E = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
#
#h1 = Hamiltonian_Legendre_polynomial(c, potential, domain, N)
#eigenvalues = np.linalg.eigvals(h1)
#eigenvalues_real_sort = np.sort(eigenvalues.real)
##selecting first five energy from our calculated results
##only keep one digit after the decimal
#E_calc = np.around(eigenvalues_real_sort[:5], decimals = 1)

#print(eigenvalues)

#print(E_calc)


