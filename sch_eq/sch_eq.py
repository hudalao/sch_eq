# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math as mt
import numpy.polynomial.legendre as legen
import cmath
#domain is the range of x
#N is the number of the sampling
#function is the input values of function at each sampling point


#here, we use the momentum basis which is a fourier basis set
#num_n is the number of functions in the basis seti
def Hamiltonian_momentum_basis(c, potential, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_x = domain / (N - 1)
    #potential term
    V = np.zeros((N, N), dtype = complex)

    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                brax_ketp = cmath.exp( exp_coeff[jj] * x[kk] )
                brap_ketx = cmath.exp( -1 * exp_coeff[ii] * x[kk] )
                add = brap_ketx * potential[kk] * brax_ketp * delta_x
                V[ii][jj] = V[ii][jj] + add
    #kinetic term
    K = np.zeros((N, N), dtype = complex)

    K_coeff = c * 4 * np.pi ** 2 / domain 
    for ii in range(N):
        K[ii][ii] = n[ii] ** 2

    K = K_coeff * K
    return (K + V) / domain

def Hamiltonian_kinetic_basis(c, potential, domain, N):
    x = np.linspace(-1 * domain / 2, domain /2, N)
    n = np.linspace(-N / 2, N / 2, N)
    coeff = 2 * np.pi * n / domain

    #potential term
    V = np.zeros((N, N))

    for ii in range(N):
        for jj in range(N):
            for kk in range(N):
                brax_ketp = mt.cos( coeff[jj] * x[kk] )
                brap_ketx = mt.cos( coeff[ii] * x[kk] )
                add = brap_ketx * potential[kk] * brax_ketp
                V[ii][jj] = V[ii][jj] + add
    
    #kinetic term
    K = np.zeros((N, N))

    K_coeff = c * 4 * np.pi ** 2 / domain ** 2
    for ii in range(N):
        K[ii][ii] = n[ii] ** 2
        inner_product = 0
        for kk in range(N):
            add = mt.cos(coeff[ii] * x[kk]) ** 2
            inner_product = inner_product + add
        K[ii][ii] = K[ii][ii] * inner_product
    K = K_coeff * K
    return K + V

#test
#using Hartree atomic units
#define c as h_bar **2 /2m
#m = 1
#omega = 1
#c = 1 / 2 / m
#
#domain = 15
#N = 70
#
#x  = np.linspace(-1 * domain / 2, domain / 2, N)
#potential = m * omega ** 2 * x ** 2 / 2
#
#h1 = Hamiltonian_momentum_basis(c, potential, domain, N)
#eigenvalues = np.linalg.eigvals(h1)
#print(np.sort(eigenvalues))


def Hamiltonian_Legendre_polynomial(c, V, domain, N, wave_func):
    
    x = np.linspace(-domain / 2, domain /2, N)
    delta_x = domain / (N - 1)
    
    #represent out wave function in the legendre polynomial basis
    wave_poly = legen.legfit(x, wave_func, N)
    
    #calculate H |bj>, where H = -c Lap + V
    
    #calculate -c Lap |bj>
    Hbj_first = legen.legder(wave_poly, 2)
    #calculate V|bj>, here, V is a constant
    Hbj_secod = V * wave_poly
    Hbj = Hbj_first + Hbj_secod

    # output = legen.legval(x, Hbj, tensor = True)
    
    return Hbj
#    #potential term
#    V = np.zeros((N, N))
#
#    for ii in range(N):
#        legen_bra = np.zeros(ii + 1)
#        legen_bra[ii] = 1
#        for jj in range(N):
#            legen_ket = np.zeros(jj + 1)
#            legen_ket[jj] = 1
#            for kk in range(N):
#                add = legen.legval(x[kk], legen_bra) * \
#                        potential[kk] * \
#                        legen.legval(x[kk], legen_ket) * delta_x
#                V[ii][jj] = V[ii][jj] + add
#    
#    #kinetic term
#
#    K = np.zeros((N, N))
#
#    for ii in range(N):
#        legen_bra = np.zeros(ii + 1)
#        legen_bra[ii] = 1
#        for jj in range(N):
#            derive_array = np.zeros(jj + 1)
#            derive_array[jj] = 1
#            legen_ket_derive = legen.legder(derive_array, 2)
#            for kk in range(N):
#                add = legen.legval(x[kk], legen_bra) * \
#                        legen.legval(x[kk], legen_ket_derive) * delta_x
#                K[ii][jj] = K[ii][jj] + add
#
#    K = K * -1 * c
#    print(K)
#    print(V)
#    return K + V

m = 1
omeca = 1
c = 1 / 2 / m

domain = 10
N = 70

x  = np.linspace(-domain / 2, domain /2, N)
potential = m * omeca ** 2 * x ** 2 / 2

h1 = Hamiltonian_Legendre_polynomial(c, potential, domain, N)
eigenvalues = np.linalg.eigvals(h1)
print(eigenvalues)
