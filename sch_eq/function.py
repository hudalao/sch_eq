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
            add = wave_func[kk] * cmath.exp( -1 * exp_coeff[ii] * x[kk] )
            a[ii] = a[ii] + add
    a = a / N
    return a

#reconstruct the original function for testing purpose
def reconstruct_wave(wave_fourier_coeff, domain, N):
    x = np.linspace(-domain / 2, domain / 2, N)
    n = np.linspace(-N / 2 + 1, N / 2, N)
    exp_coeff = 1j * 2 * np.pi * n / domain
    delta_p = 2 * np.pi / domain
    wave = np.zeros(N, dtype = complex)
    for kk in range(N):
        for ii in range(N):
            add = wave_fourier_coeff[ii] * \
                    cmath.exp( exp_coeff[ii] * x[kk] )
            wave[kk] = wave[kk] + add
    
    wave = wave * delta_p
    
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
    
    #it is known that HC = HSC, because S here is a identity matrix with elements 
    # equals to period, we can just divide the H by period value
    H = (K + V) / domain

    return H


def Legendre_polynomial_basis(c, potential, domain, N, wave_func):
    
    x = np.linspace(-domain / 2, domain /2, N)
    
    #represent out wave function in the legendre polynomial basis
    wave_legen = legen.legfit(x, wave_func, N)
    
    #calculate H |bj>, where H = -c Lap + V
    
    #calculate -c Lap |bj>
    Hbj_first = -1 * c * legen.legder(wave_legen, 2)
    #calculate V|bj>, here, V is a constant
    Hbj_secod = potential * wave_legen
    Hbj = Hbj_first + Hbj_secod[0: N - 1]
    
    return Hbj

def Hamiltonian_Legendre_polynomial(c, potential, domain, N):
    #potential is a constant in this case

    x = np.linspace(-domain / 2, domain /2, N)
    delta_x = domain / (N - 1)
    
    #here, the normalized legendre polynomical has been used 
    # for the nth polynomials, normalization constant is sqrt(2/(2n + 1))
    
    #kinetic term
    K = np.zeros((N, N))

    for ii in range(N):
        legen_left = np.zeros(N)
        legen_left[ii] = mt.sqrt((2 * ii + 1) / 2) 
        for jj in range(N):
            deriva_array = np.zeros(N + 2)
            deriva_array[jj] = mt.sqrt((2 * jj + 1) / 2)
            legen_right_deriva = legen.legder(deriva_array, 2)
            
            #multiply them
            legen_multiply = legen.legmul(legen_left, legen_right_deriva)
            
            #integral
            legen_integral = legen.legint(legen_multiply)
            
            #calculate the matrix elements
            K[ii][jj] = legen.legval(domain / 2, legen_integral) - \
                        legen.legval(-domain / 2, legen_integral)
           
    #the S matrix, inside the [-1, 1] domain, the legendre ploynomial can be treatedas basis and satisfying <xi|xj> = delta ij, thus S matrix is a identity matrix
    S = np.zeros((N, N))
    
    for ii in range(N):
        legen_left_S = np.zeros(N)
        legen_left_S[ii] = mt.sqrt((2 * ii + 1) / 2)
        legen_multiply_S = legen.legmul(legen_left_S, legen_left_S)
        legen_integral_S = legen.legint(legen_multiply_S)
        S[ii][ii] = legen.legval(domain / 2, legen_integral_S) - \
                    legen.legval(-domain / 2, legen_integral_S)

    K = K * -1 * c
   
    #because the potential is just a constant here, we can calculate the V matrix   simply by multiply the matrix S a constant potential value
    
    V = potential * S
    
    ##divide the obtained Hamiltonian by the S matrix
    H = K + V
    return H

