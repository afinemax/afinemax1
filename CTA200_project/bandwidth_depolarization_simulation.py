# -*- coding: utf-8 -*-
"""
Created on Sat May 16 2020 CE

@author: Maxwell A. Fine

Made for CITA200h Computational Astrophysics, and

SURP: Summer Undergaduate Research Program

Supervisor: Dr. Cameron van Eck
"""


# modeification of ex1-10 for ex11
import numpy as np
from scipy.constants import speed_of_light
from scipy.integrate import quad
import scipy as sp
import matplotlib.pyplot as plt

c = speed_of_light

xi_knot = 1
p = 1


def frequency_spaceing(lower, upper, n):
    '''Generates a 1D array of equally spaced frequency values (in Hz)
    
    Args:
    lower = lower bound frequency for array (inclusive)
    upper = upper bound frequency for arary (inclusive)
    n = number of samples in the array
    
    Returns:
    frequency_array = a 1D array of equally spaced frequency values (in Hz)'''
    
    frequency_array = np.linspace(lower, upper, n)
    
    return frequency_array

def bandwidth(f_array):
    '''Returns bandwidth per channel of a frequency array'''
    
    ban = f_array[1] - f_array[0]
    
    return ban

def half_max(f, ban):
    '''Finds the magnitude of faraday depth for intensity to drop to half
    
    Args: 
    f = channel center frequency (in Hz)
    ban = bandwidth (in Hz)
    
    Returns:
    Float of the faraday depth value of half max'''
    
    fa = (f - 0.5 * ban)**2
    fb = (f + 0.5 * ban)**2
    
    xi_knot = 1
    p = 1
    
    predicted = int(round(np.abs(np.sqrt(3) / (c**2 *((1/fa) - (1/fb))))))
    
    margin = int(round((ban / 2)))
    
    phi = np.linspace(predicted - margin , predicted + margin, int(0.5 *round(predicted)))
    num_pol = 1 * phi
    
    n = len(phi)
    epsilon = 0.005
    for i in range(n):
        num_pol[i] = np.abs((bandwidth_avg_polarization(f, ban, phi[i], xi_knot, p)))
        if np.abs(num_pol[i] - 0.50) < epsilon:
            half_max = phi[i]
    return half_max

def bandwidth_avg_polarization(f, ban, phi, xi_knot, p):
    '''Computes the bandwidth averaged complex polarization of a single frequency channel
    
    Args:
    f = channel center frequency (in Hz)
    ban = bandwidth (in Hz)
    phi =  faraday depth value (in rad/m2)
    xi_knot = initial polarization angle (in rad)
    p = polarized intensity 
    
    Returns:
    avg_p_tilda = the average complex polarization, for the bandwidth, real is Q, imaginary is U
    '''
    a = f - (ban / 2)
    b = f + (ban / 2) # integral start and stop values
    
    x = f
    
    def func_n1(x, phi, xi_knot, p):
        return  np.real(p * np.exp(2.0j *(xi_knot + (c / x)**2 * phi ))) # integrand
    
    def func_n2(x, phi, xi_knot, p):
        return np.imag(p * np.exp(2.0j *(xi_knot + (c / x)**2 * phi )))
    
    i1 = quad(func_n1, a, b, args=(phi, xi_knot, p))[0] # integral
    
    i2 = quad(func_n2, a, b, args=(phi, xi_knot, p))[0]
    
    i = i1 + 1.0j*i2
    
    avg_p_tilda = i / ban # mean value thm
    
    return avg_p_tilda

def bandwidth_avg_array(f, phi, xi_knot, p):
    ''' computes the bandwidth averaged polarization for an array of channels

    Args:
    f = a 1D array of equally spaced frequency values (in Hz)
    phi =  faraday depth value (in rad/m2)
    xi_knot = initial polarization angle (in rad)
    p = polarized intensity
    
    Returns:
    avg_p_tilda = an array of the average complex polarization for each channel, real is Q, imaginary is U
    '''
    avg_p_tilda = 1.0j * f
    ban = bandwidth(f)
    n = len(f)
    for i in range(n): 
        avg_p_tilda[i] = bandwidth_avg_polarization(f[i], ban, phi, xi_knot, p)
    #
    
    return avg_p_tilda

def plot_stokes_intensity_angle(f, phi, xi_knot, p):
    '''Plots stokes Q and U, polarized intensity and polarized angle
    
    Args:
    f = an array of frequency values (in Hz)
    phi =  faraday depth value (in rad/m2)
    xi_knot = initial polarization angle (in rad)
    p = polarized intensity
    
    Returns:
    3 plots'''
    
    # plot 1 stokes q/u seperatly on y axis, freq on x
    p_tilda = bandwidth_avg_array(f, phi, xi_knot, p)
    
    q = np.real(p_tilda)
    u = np.imag(p_tilda)
    
    plt.figure()
    plt.plot(f, q, label='Stokes Q')
    plt.plot(f, u, label='Stokes U')
    plt.xlabel('Frequency')
    plt.ylabel('Intensity')
    plt.title('Stokes Q & U')
    plt.legend()
    plt.savefig('better_stokes_sim.pdf', dpi=400)
    plt.show()
    
    # plot 2 polarized intensity on y, freq on x
    
    p = np.abs(p_tilda)
    
    plt.figure()
    plt.plot(f, p, label='Polarized Intensity')
    plt.xlabel('Frequency')
    plt.ylabel('Intensity')
    plt.title('Polarized Intensity')
    plt.legend()
    plt.savefig('polarized_intensisty_sim.pdf', dpi=400)
    plt.show()
   
    return

def high_plot_bandwidth_depolarization(f, ban):
    '''Plots the polarized intensity for the channel as a function of increasing faraday depth
    
    Args: 
    f = channel center frequency (in Hz)
    ban = bandwidth (in Hz)
    
    Returns:
    Plot of the polarized intensity as a function of increasing faraday depth'''
    
    #analytic
    
    #phi = np.linspace(0, 30000, 30000)
    
   # num_pol = 1 * phi
   
    fa = (f - 0.5 * ban)**2
    fb = (f + 0.5 * ban)**2
    
    xi_knot = 1
    p = 1
    
    predicted_phi = np.abs(np.sqrt(3) / (c**2 *((1/fa) - (1/fb))))
    
    phi = np.linspace(-10 * int(round(predicted_phi)), 10 *int(round(predicted_phi)), int(0.5 *round(predicted_phi)))
    num_pol = 1 * phi
    
    epsilon = 0.005
    n = len(phi)
    for i in range(n):
        num_pol[i] = np.abs((bandwidth_avg_polarization(f, ban, phi[i], xi_knot, p)))
        if np.abs(num_pol[i] - 0.50) < epsilon:
            half_max = np.abs(phi[i])
    
    plt.figure()
    plt.plot(phi, num_pol,'b')
    #plt.plot(phi, an_pol,'r',label='Analytical')
    plt.xlabel('Faraday Depth')
    plt.ylabel('Fractional Intensity')
    plt.vlines(half_max, 0, 1.0, colors='purple', linestyles='dashed', label='Faraday Depth of Half Max')
    plt.vlines(-half_max, 0, 1.0, colors='purple', linestyles='dashed',)
    plt.hlines(0.5, -10 * int(round(predicted_phi)), 10 * int(round(predicted_phi)), colors='black', linestyles='dashed', label='Half max')
    plt.title('Highest Frequency Bandwidth Depolarization')
    #plt.legend('upper left')
    plt.savefig('high_bandwidth_depolarization_sim.pdf', dpi=400)
    plt.show()
    
    return 

def low_plot_bandwidth_depolarization(f, ban):
    '''Plots the polarized intensity for the channel as a function of increasing faraday depth
    
    Args: 
    f = channel center frequency (in Hz)
    ban = bandwidth (in Hz)
    
    Returns:
    Plot of the polarized intensity as a function of increasing faraday depth'''
    
    #analytic
    
    #phi = np.linspace(0, 30000, 30000)
    
   # num_pol = 1 * phi
   
    fa = (f - 0.5 * ban)**2
    fb = (f + 0.5 * ban)**2
    
    xi_knot = 1
    p = 1
    
    predicted_phi = np.abs(np.sqrt(3) / (c**2 *((1/fa) - (1/fb))))
    
    phi = np.linspace(-10 * int(round(predicted_phi)), 10 *int(round(predicted_phi)), int(0.5 *round(predicted_phi)))
    num_pol = 1 * phi
    
    epsilon = 0.005
    n = len(phi)
    for i in range(n):
        num_pol[i] = np.abs((bandwidth_avg_polarization(f, ban, phi[i], xi_knot, p)))
        if np.abs(num_pol[i] - 0.50) < epsilon:
            half_max = np.abs(phi[i])
    
    plt.figure()
    plt.plot(phi, num_pol,'b')
    #plt.plot(phi, an_pol,'r',label='Analytical')
    plt.xlabel('Faraday Depth')
    plt.ylabel('Fractional Intensity')
    plt.vlines(half_max, 0, 1.0, colors='purple', linestyles='dashed', label='Faraday Depth of Half Max')
    plt.vlines(-half_max, 0, 1.0, colors='purple', linestyles='dashed',)
    plt.hlines(0.5, -10 * int(round(predicted_phi)), 10 * int(round(predicted_phi)), colors='black', linestyles='dashed', label='Half max')
    plt.title('Lowest Highest Frequency Bandwidth Depolarization')
    #plt.legend('upper left')
    plt.savefig('low_bandwidth_depolarization_sim.pdf', dpi=400)
    plt.show()
    
    return 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("lower", help="lower bound frequency for array (inclusive)",
                    type=int)
parser.add_argument("upper", help="upper = upper bound frequency for arary (inclusive)",
                    type=int)
parser.add_argument("n", help="number of samples in the array",
                    type=int)
args = parser.parse_args()


lower = args.lower
upper = args.upper 
n = args.n

f = frequency_spaceing(lower, upper, n)

ban = bandwidth(f)
    

high = f[0]
low = f[-1]
phi = half_max(low, ban)
high_plot_bandwidth_depolarization(high, ban)
low_plot_bandwidth_depolarization(low, ban)
plot_stokes_intensity_angle(f, phi, xi_knot, p)


