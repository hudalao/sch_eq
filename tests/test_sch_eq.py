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

from sch_eq.sch_eq import output
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
        self.assertEqual(E.all(), E_calc.all())
        
    def test_Legendre_polynomial(self):
        #set V = 0, and input wavefunction is a sin function, using the column              vector we obtained in legendre polynomial basis, we should restruct the             sin function
        pass
