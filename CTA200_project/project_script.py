# -*- coding: utf-8 -*-
"""
Created on Sat May 16 16:15:05 2020

@author: Max Fine
"""

import numpy as np
from scipy.constants import speed_of_light
from scipy.integrate import quad
import scipy as sp
import matplotlib.pyplot as plt

c = speed_of_light


# ex2
def frequency_spaceing(lower, upper, n):
    '''Generates a 1D array of equally spaced frequency values (in Hz)
    
    Args:
    lower = lower bound frequency for array (inclusive)
    upper = upper bound frequency for arary (inclusvive)
    n = number of samples in the array
    
    Returns:
    frequency_array = a 1D array of equally spaced frequency values (in Hz)'''
    
    frequency_array = np.linspace(lower, upper, n)
    
    return frequency_array

# ex3
def stokes_q_u(f, phi, xi_knot, p):
    '''Calculates Stokes Q and U as a function of frequency
    
    Args:
    f = an array of frequency values (in Hz)
    phi =  faraday depth value (in rad/m2)
    xi_knot = initial polarization angle (in rad)
    p = polarized intensity 
    
    Returns:
    p_tilda = single complex array of complex polarization, real is Q, imaginary is U
    '''

    p_tilda = p * np.exp(2.0j *(xi_knot + (c / f)**2 * phi ))
    
    return p_tilda

