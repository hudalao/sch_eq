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

#domain is the range of x and V(x)
#c the constant
#N the size of the basis set
#V the potential energy V(x) ps: the size of V(x) should be same as the size of the basis set
#V_const the constant potential energy 
#the choice of basis set function: 1 ---> the fourier basis 2 ---> the legendre polynomial basis
#ps: the fourier basis can take function V of x, but the legendre polynomial basis  can only take the constant V. Be careful when you use different basis method


#with input wave function, calculate its coefficient under the fourier basis
def wave_fourier_basis(wave_func, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_x = domain / (N - 1)
    a = np.zeros(N, dtype = complex)
    for ii in range(1, N):
        for kk in range(N):
            add = wave_func[kk] * cmath.exp( -1 * exp_coeff[ii] * x[kk] ) * delta_x
            a[ii] = a[ii] + add
    a = a / domain
    return a

#reconstruct the original function for testing purpose
def reconstruct_wave(wave_fourier_coeff, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_x = domain / (N - 1)
    wave = np.zeros(N, dtype = complex)
    for kk in range(N):
        for ii in range(N):
            add = wave_fourier_coeff[ii] * \
                    cmath.exp( exp_coeff[ii] * x[kk] ) * delta_x
            wave[kk] = wave[kk] + add
    wave = wave * domain  
    return wave
    
#here, we use the momentum basis which is a fourier basis set, which means we reprsent the whole (-c Lap + V) as matrix with the momentum basis
#potential here refers to V in the equation shown above
#the reson using this method is that we can obtain the eigenvalues and eigenvectors directly by diaglize this matrix
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

    K_coeff = c * 4 * np.pi ** 2 / domain ** 2 
    for ii in range(N):
        K[ii][ii] = n[ii] ** 2 * N * delta_x

    K = K_coeff * K
    return (K + V) / domain 


def Legendre_polynomial_basis(c, V, domain, N, wave_func):
    
    x = np.linspace(-domain / 2, domain /2, N)
    
    #represent out wave function in the legendre polynomial basis
    wave_poly = legen.legfit(x, wave_func, N)
    
    #calculate H |bj>, where H = -c Lap + V
    
    #calculate -c Lap |bj>
    Hbj_first = -1 * c * legen.legder(wave_poly, 2)
    #calculate V|bj>, here, V is a constant
    Hbj_secod = V * wave_poly
    Hbj = Hbj_first + Hbj_secod[0: N - 1]
    
    return Hbj
