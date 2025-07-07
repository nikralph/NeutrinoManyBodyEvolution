import numpy as np
import matplotlib.pyplot as plt

from scipy.special import comb

# Time Evolution Operator; Returns probability
def timeywimey(EigValues, EigVectors, state, timey):
    for i in range(len(EigValues)):
        EigValues[i] = np.exp(-1j*EigValues[i]*timey)
    EigValues = np.diag(EigValues)
    expectation = state.T@(EigVectors@EigValues@np.linalg.inv(EigVectors))@state
    probability = expectation*np.conjugate(expectation)
    return probability[0,0]

# Graph data generator for time evolution operator expectation values
def timeyGraphdata(EigValues, EigVectors, state, timey, resolution):
    times = np.zeros(resolution, dtype=complex)
    ExpEigValues = np.zeros(resolution, dtype=complex)
    for i in range(resolution):
        times[i] = (i*timey)/resolution
        ExpEigValues[i] = timeywimey(np.copy(EigValues), EigVectors, state, times[i])
    return times, ExpEigValues

# Returns information for a momentum mode
def Pinfo(p):
    pabs = np.sqrt(np.sum(p*p))
    ftheta = np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2])
    fphi = np.arctan2(p[1], p[0])
    return pabs, ftheta, fphi

# gfactor for Hvv blocks, Fp1p2 is Fij-dagger(p1, p2) and Fq1q2 is Fij(q1, q2)
def gfactor(p1, p2, q1, q2):
    p1abs, p1theta, p1phi = Pinfo(p1)
    p2abs, p2theta, p2phi = Pinfo(p2)
    q1abs, q1theta, q1phi = Pinfo(q1)
    q2abs, q2theta, q2phi = Pinfo(q2)
    Fp1p2 = np.exp(1j*p1phi)*np.sin(p1theta/2)*np.cos(p2theta/2) - np.exp(1j*p2phi)*np.sin(p2theta/2)*np.cos(p1theta/2)
    Fq1q2 = np.exp(-1j*q1phi)*np.sin(q1theta/2)*np.cos(q2theta/2) - np.exp(-1j*q2phi)*np.sin(q2theta/2)*np.cos(q1theta/2)
    return 2*Fp1p2*Fq1q2

# Generate grid of momentum modes and check conservation of P and KE. Returns Pstates, 
def PGenerator(zmax):
    Pstates = []
    zxrange = np.array(range(1,zmax+1))
    zyrange = np.array(range(-zmax+1,zmax))
    for x in zxrange:
        for y in zyrange:
            if x**2 + y**2 <= zmax**2:
                Pstates.append([x,y,0])
    Pstates = np.array(Pstates)
    
    Nps = len(Pstates)
    
    for ip1 in range(Nps):
        p1 = Pstates[ip1,:]
        for ip2 in range(i1,Nps):
            p2 = Pstates[ip2,:]
            for iq1 in range(Nps):
                q1 = Pstates[iq1,:]
                for iq2 in range(j1,Nps):
                    q2 = Pstates[iq2,:]
                    Pconserve = p1+p2-q1-q2
                    KEconserve = np.linalg.norm(p1)+np.linalg.norm(p2)-np.linalg.norm(q1)-np.linalg.norm(q2)
                    if np.linalg.norm(Pconserve) < 1e-9 and KEconserve < 1e-9:
                        
    
    return

# Construct the Hamiltonian. Returns both Full (F) and Truncated (T) versions.
def buildHamiltonian():
    # Initialize Hamiltonians
    H = np.zeros((dim, dim), dtype=complex)
    HT = np.zeros((dim, dim), dtype=complex)
    Hkin = np.zeros((dim, dim), dtype=complex)
    Hvv = np.zeros((dim, dim), dtype=complex)
    HTvv = np.zeros((dim, dim), dtype=complex)
    return





# Formatting
np.set_printoptions(formatter={'all': lambda x: "{:.12g}".format(x)})

# 
try:
    Nnu = input("Enter the number of neutrinos: ")      # Number of Neutrinos
    Nflav = input("Enter the number of flavors: ")      # Number of Flavor states
    zmax = input("Enter the momentum threshold: ")







