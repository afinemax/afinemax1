#=============================================================================#
#                          MODEL DEFINITION FILE                              #
#=============================================================================#
import numpy as np
from scipy.constants import speed_of_light
from scipy.integrate import quad
import scipy as sp
import matplotlib.pyplot as plt

c = speed_of_light


#-----------------------------------------------------------------------------#
# Function defining the model.                                                #
#                                                                             #
#  pDict       = Dictionary of parameters, created by parsing inParms, below. #
#  lamSqArr_m2 = Array of lambda-squared values                               #
#  quArr       = Complex array containing the Re and Im spectra.              #
#-----------------------------------------------------------------------------#
def analytical(a, phi, xi_knot):
    '''computes the bound values for the wolfram alpha integral'''
    func4 = 0
    
    funct1 = a * np.exp((2.0j * phi * c**2) / (a**2))
    funct2 = -(1 + 1.0j)*np.sqrt(np.pi)*np.sqrt(phi)*c
    funct3 = sp.special.erfi(((1 + 1.0j) * np.sqrt(phi) * c)/a)
    func4 =  np.exp(2.0j*xi_knot)
    ya = (funct1 + (funct2 * funct3))*func4
    
    return ya

def analytic_solution_polarization_integral_channel(f, ban, phi, xi_knot, p):
    '''Calculates the average analytic solution to the channel polarization integral for 1 channel
    
    Based on equation 13 of Schnitzeler & Lee (2015)
    
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
                   
    ya =  analytical(a, phi)
    yb =  analytical(b, phi)
                     
    i = yb - ya
    avg_p_tilda = i / ban
    
    return avg_p_tilda

def bandwidth(f_array):
    '''Returns bandwidth per channel of a frequency array'''
    
    ban = f_array[1] - f_array[0]
    
    return ban


def model(pDict, lamSqArr_m2):
    """Two separate Faraday components, averaged within same telescope beam
    (i.e., unresolved), with individual Burn depolarisation terms."""
    
    # Calculate the complex fractional q and u spectra
    #pArr1 = pDict["fracPol1"] * np.ones_like(lamSqArr_m2)
    #pArr2 = pDict["fracPol2"] * np.ones_like(lamSqArr_m2) # array of frac pol
    #quArr1 = pArr1 * np.exp( 2j * (np.radians(pDict["psi01_deg"]) +
                                   #pDict["RM1_radm2"] * lamSqArr_m2))
    #quArr2 = pArr2 * np.exp( 2j * (np.radians(pDict["psi02_deg"]) +
                                  # pDict["RM2_radm2"] * lamSqArr_m2))
    
    #quArr = (quArr1 * np.exp(-2.0 * pDict["sigmaRM1_radm2"]**2.0
                            # * lamSqArr_m2**2.0) +
            # quArr2 * np.exp(-2.0 * pDict["sigmaRM2_radm2"]**2.0
                           #  * lamSqArr_m2**2.0))
    
    f = np.sqrt(c**2 / lamSqArr_m2)  # only need 1 Parr1
   
    psi_knot1 = np.radians(pDict["psi01_deg"])
    phi1 = pDict["RM1_radm2"]
                          
    ban = bandwidth(f)
    
    quArr1 = analytic_solution_polarization_integral_channel(f, ban, phi1, psi_knot1, pDict["fracPol1"])
    
    quArr = quArr1
                  
    return quArr


#-----------------------------------------------------------------------------#
# Parameters for the above model.                                             #
#                                                                             #
# Each parameter is defined by a dictionary with the following keywords:      #
#   parname    ...   parameter name used in the model function above          #
#   label      ...   latex style label used by plotting functions             #
#   value      ...   value of the parameter if priortype = "fixed"            #
#   bounds     ...   [low, high] limits of the prior                          #
#   priortype  ...   "uniform", "normal", "log" or "fixed"                    #
#   wrap       ...   set > 0 for periodic parameters (e.g., for an angle)     #
#-----------------------------------------------------------------------------#
inParms = [
    {"parname":   "fracPol1",
     "label":     "$p_1$",
     "value":     0.1,
     "bounds":    [0.001, 1.0],
     "priortype": "uniform",
     "wrap":      0},
    
    
    {"parname":   "psi01_deg",
     "label":     "$\psi_{0,1}$ (deg)",
     "value":     0.0,
     "bounds":    [0.0, 180.0],
     "priortype": "uniform",
     "wrap":      1},
    

    
    {"parname":   "RM1_radm2",
     "label":     "$\phi_1$ (rad m$^{-2}$)",
     "value":     0.0,
     "bounds":    [-1100.0, 1100.0],
     "priortype": "uniform",
     "wrap": 0},
    
]


#-----------------------------------------------------------------------------#
# Switches controlling the Nested Sampling algorithm                          #
#-----------------------------------------------------------------------------#
nestArgsDict = {"n_live_points": 1000,
                "verbose": False}

