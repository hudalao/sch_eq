#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_sch_eq
----------------------------------

Tests for `sch_eq` module.
"""

import sys
import unittest

import numpy as np
import matplotlib.pyplot as plt
import math as mt
import numpy.polynomial.legendre as legen
import cmath

from sch_eq.sch_eq import output, ground_wave_function
from sch_eq.function import wave_fourier_basis, Hamiltonian_momentum_basis, \
        Legendre_polynomial_basis, reconstruct_wave


class TestSch_eq(unittest.TestCase):
    
    def test_harmonic_oscillator(self):
        #the ODE describing the harmonic oscillator has the same for as out ODE,            becaue we know the eigenvalues(energy) of the harmonic oscillator                   E = h_bar * frequence * (N + 1/2)   where N is the occupation number of             phonons, the values of N is from 0 to infinity
        
        #using Hartree atomic units, h_bar = 1
        #define c as h_bar ** 2 /2m
        m = 1
        omega = 1
        c = 1 / 2 / m 
        domain = 15
        N = 70
        
        #the corresponding potential
        x  = np.linspace(-1 * domain / 2, domain / 2, N)
        potential = m * omega ** 2 * x ** 2 / 2
        
        #from analytic solution, the first five energy are
        E = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        
        h1 = Hamiltonian_momentum_basis(c, potential, domain, N)
        eigenvalues = np.linalg.eigvals(h1)
        eigenvalues_real_sort = np.sort(eigenvalues.real)
        #selecting first five energy from our calculated results
        #only keep one digit after the decimal
        E_calc = np.around(eigenvalues_real_sort[:5], decimals = 1)
        for ii in range(5):
            self.assertEqual(E[ii], E_calc[ii])
        
    def test_Legendre_polynomial_after_operating_wave_function(self):
        #set V = 0, and input wavefunction is a sin function, using the column vector we obtained after operating the input wavefunction with Hamiltonian represented by legendre polynomials, we should restruct the sin function
        N = 50
        domain = 2 * np.pi
        x = np.linspace(-domain / 2,domain / 2, N)
        
        wave_func = np.sin(x)
        potential = np.repeat(0, N)
        
        #Legen_polynomial
        c2 = output(1, potential, 0, N, wave_func, 2, domain)
        a2 = legen.legval(x, c2)
        
        a2 = np.round(a2, decimals = 2)
        wave_func = np.round(wave_func, decimals = 2)

        for ii in range(N):
            self.assertEqual(a2[ii], wave_func[ii])

    def test_fourier_basis_after_operating_wave_function(self):
        #set V = 0, and input wavefunction is a sin function, using the column vector we obtained after operating the input wavefunction with Hamiltonian in fourier basis, we should restruct the sin function
        N = 50
        domain = 2 * np.pi
        x = np.linspace(-domain / 2,domain / 2, N)
        
        wave_func = np.sin(x)
        potential = np.repeat(0, N)
        
        #fourier basis
        c1 = output(1, potential, 0, N, wave_func, 1, domain)
        a1 = reconstruct_wave(c1, domain, N)
        a1 = a1.real
        
        a1 = np.round(a1, decimals = 2)
        wave_func = np.round(wave_func, decimals = 2)

        for ii in range(N):
            self.assertEqual(a1[ii], wave_func[ii])
    
    def test_case(self):
        N = 50
        domain = 2 * np.pi
        x = np.linspace(-domain / 2,domain / 2, N)
        potential = np.repeat(0, N)
        wave_func = np.sin(x)
        
        output(1, potential, 0, N, wave_func, 1, domain)
        output(1, potential, 0, N, wave_func, 1, domain)

        ground_wave_function(1, potential, 0, domain, N, 1)
        ground_wave_function(1, potential, 0, domain, N, 2)

