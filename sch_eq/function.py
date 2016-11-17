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
    
    #it is known that HC = HSC
    H = (K + V) / domain

    
    ##the S matrix
    #S = np.zeros((N, N), dtype = complex)
    #
    #for ii in range(N):
    #    for jj in range(N):
    #        for kk in range(N):
    #            brax_ketp = cmath.exp( exp_coeff[jj] * x[kk] )
    #            brap_ketx = cmath.exp( -1 * exp_coeff[ii] * x[kk] )
    #            add = brap_ketx * brax_ketp * delta_x
    #            S[ii][jj] = S[ii][jj] + add



    #divide the obtained Hamiltonian by the S matrix
    #H = np.divide(K + V, S)
    return H


def Legendre_polynomial_basis(c, domain, N, wave_func):
    
    x = np.linspace(-domain / 2, domain /2, N)
    
    #represent out wave function in the legendre polynomial basis
    wave_legen = legen.legfit(x, wave_func, N - 1)
    
    ##calculate H |bj>, where H = -c Lap + V
    #
    ##calculate -c Lap |bj>
    #Hbj_first = -1 * c * legen.legder(wave_poly, 2)
    ##calculate V|bj>, here, V is a constant
    #Hbj_secod = V * wave_poly
    #Hbj = Hbj_first + Hbj_secod[0: N - 1]
    return wave_legen

def Hamiltonian_Legendre_polynomial(c, potential, domain, N):
    
    x = np.linspace(-domain / 2, domain /2, N)
    delta_x = domain / (N - 1)
    
    ##potential term
    #V = np.zeros((N, N))
    #
    ##potential_legen = legen.legfit(x, potential, N - 1)
    #
    #for ii in range(N):
    #    legen_left_V = np.zeros(N)
    #    legen_left_V[ii] = 1
    #    for jj in range(N):
    #        legen_right_V = np.zeros(N)
    #        #legen_right_V[jj] = potential_legen[jj]
    #        legen_right_V[jj] = 1
    #        
    #        ##multiply 
    #        #legen_multiply_V = legen.legmul(legen_left_V, legen_right_V)
    #       
    #        ##integral
    #        #legen_integral_V = legen.legint(legen_multiply_V)
    #       
    #        ##calculate the matrix elements
    #        #V[ii][jj] = legen.legval(domain / 2, legen_integral_V) - \
    #        #            legen.legval(-domain / 2, legen_integral_V)
    #        for kk in range(N):
    #            add = legen.legval(x[kk], legen_left_V) * \
    #                    potential[kk] * legen.legval(x[kk], legen_right_V)
    #            V[ii][jj] = V[ii][jj] + add
    #V = V * delta_x
    
    #kinetic term
    K = np.zeros((N, N))
    
    for ii in range(N):
        legen_left = np.zeros(N)
        legen_left[ii] = 1
        for jj in range(N):
            deriva_array = np.zeros(N + 2)
            deriva_array[jj] = 1
            legen_right_deriva = legen.legder(deriva_array, 2)
            
            #multiply them
            legen_multiply = legen.legmul(legen_left, legen_right_deriva)
            
            #integral
            legen_integral = legen.legint(legen_multiply)
            
            #calculate the matrix elements
            K[ii][jj] = legen.legval(domain / 2, legen_integral) - \
                        legen.legval(-domain / 2, legen_integral)
           
    #the S matrix
    S = np.zeros((N, N))
    
    for ii in range(N):
        legen_left_S = np.zeros(N)
        legen_left_S[ii] = 1
        #for jj in range(N):
        #    legen_right_S = np.zeros(N)
        #    legen_right_S[jj] = 1
        #    
        #    #multiply 
        #    legen_multiply_S = legen.legmul(legen_left_S, legen_right_S)
        #   
        #    #integral
        #    legen_integral_S = legen.legint(legen_multiply_S)
        #   
        #    #calculate the matrix elements
        #    S[ii][jj] = legen.legval(domain / 2, legen_integral_S) - \
        #                legen.legval(-domain / 2, legen_integral_S)
        legen_multiply_S = legen.legmul(legen_left_S, legen_left_S)
        legen_integral_S = legen.legint(legen_multiply_S)
        S[ii][ii] = legen.legval(domain / 2, legen_integral_S) - \
                    legen.legval(-domain / 2, legen_integral_S)

    K = K * -1 * c
    
    V = potential * S
    
    print(S)
    #divide the obtained Hamiltonian by the S matrix
    S_inverse = np.linalg.inv(S)
    H = np.dot(S_inverse, K + V)

    return H

