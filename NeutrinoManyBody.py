# This program is based on the paper https://arxiv.org/pdf/2404.16690
# Thank you Vincenzo Cirigliano and Yukari Yamauchi!

import numpy as np
import matplotlib.pyplot as plt
#import sys

from itertools import combinations

# Time Evolution Operator; Returns probability
def timeywimey(EigValues, EigVectors, state, timey):
    for i in range(len(EigValues)):
        EigValues[i] = np.exp(-1j*EigValues[i]*timey)
    EigValues = np.diag(EigValues)
    expectation = state.T@(EigVectors@EigValues@np.linalg.inv(EigVectors))@state
    probability = expectation*np.conjugate(expectation)
    return probability

# Graph data generator for time evolution operator expectation values
def timeyGraphdata(EigValues, EigVectors, state, timey, resolution):
    times = np.zeros(resolution, dtype=complex)
    ExpEigValues = np.zeros(resolution, dtype=complex)
    for i in range(resolution):
        times[i] = (i*timey)/resolution
        ExpEigValues[i] = timeywimey(np.copy(EigValues), EigVectors, state, times[i])
    return times, ExpEigValues

# Returns information for a momentum mode
def pinfo(p):
    pabs = np.sqrt(np.sum(p*p))
    ftheta = np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2])
    fphi = np.arctan2(p[1], p[0])
    return pabs, ftheta, fphi

# gfactor for Hvv blocks, Fp1p2 is Fij-dagger(p1, p2) and Fq1q2 is Fij(q1, q2)
def gfactor(p1, p2, q1, q2):
    p1abs, p1theta, p1phi = pinfo(p1)
    p2abs, p2theta, p2phi = pinfo(p2)
    q1abs, q1theta, q1phi = pinfo(q1)
    q2abs, q2theta, q2phi = pinfo(q2)
    Fp1p2 = np.exp(1j*p1phi)*np.sin(p1theta/2)*np.cos(p2theta/2) - np.exp(1j*p2phi)*np.sin(p2theta/2)*np.cos(p1theta/2)
    Fq1q2 = np.exp(-1j*q1phi)*np.sin(q1theta/2)*np.cos(q2theta/2) - np.exp(-1j*q2phi)*np.sin(q2theta/2)*np.cos(q1theta/2)
    return 2*Fp1p2*Fq1q2

# Generate grid of momentum modes and check conservation of P and KE.
# Returns the momentum modes and the allowed two body transitions, as well as their g-factors.
def pGenerator(zmax):
    print("Generating momentum modes...")
    Pstates = []
    # 1 <= x <= zmax and -zmax+1 <= y <= zmax-1 represents integer coordinates on the circle of radius 5, with a positive x value.
    zxrange = np.array(range(1,zmax+1))
    zyrange = np.array(range(-zmax+1,zmax))
    # Can add in a zzrange if desired, to extend to three dimensional system
    for x in zxrange:
        for y in zyrange:
            if x**2 + y**2 <= zmax**2:
                Pstates.append([x,y,0])
    Pstates = np.array(Pstates)
    Nps = len(Pstates)
    print(f"Number of momentum modes: {Nps}")
    print("Generating list of two body interactions given conservation of P and KE...")
    pkectrans = []
    gfs = []
    momenta4 = []
    for ip1 in range(Nps):
        p1 = Pstates[ip1,:]
        for ip2 in range(Nps):
            p2 = Pstates[ip2,:]
            for iq1 in range(Nps):
                q1 = Pstates[iq1,:]
                for iq2 in range(Nps):
                    q2 = Pstates[iq2,:]
                    Pconserve = p1+p2-q1-q2
                    if np.linalg.norm(Pconserve) < 1e-9:
                        KEconserve = np.linalg.norm(p1)+np.linalg.norm(p2)-np.linalg.norm(q1)-np.linalg.norm(q2)
                        if np.abs(KEconserve) < 1e-9:
                            gf = gfactor(p1,p2,q1,q2)
                            if np.abs(gf) > 1e-9:
                                momenta4.append([ip1,ip2,iq1,iq2])
                                gfs.append(gf)
                                if ip1 <= ip2 and iq1 <= iq2 and ip1 != iq1:
                                    pkectrans.append([ip1,ip2,iq1,iq2])
    pkectrans = np.array(pkectrans)
    momenta4 = np.array(momenta4)
    gfs = np.array(gfs)
    print(f"Number of unique 4-momenta transitions: {len(pkectrans)}")
    print(f"Number of all 4-momenta transitions: {len(momenta4)}")
    return Pstates, Nps, pkectrans, momenta4, gfs

# Check if a momentum mode is used more than Nflav times
def check(state, Nflav):
    truth = True
    for i in range(len(state)-Nflav)
        if np.var(state[i:i+1+Nflav]) < 1e-9:
            truth = False
    return truth

# For a given pair of Nnu momentum modes, return the pairs that Hvv can take to
def apply(state, Nflav, Nbinom, binom, pkectrans):
    newstate = []
    for i in range(Nbinom):
        k = np.array([state[binom[i,0]], state[binom[i,1]]])
        for j in range(len(pkectrans)):
            if np.sum(np.abs(k-pkectrans[j,:2]))==0:
                state_i = state.copy()
                state_i[binom[i,0]] = pkectrans[j,2]
                state_i[binom[i,1]] = pkectrans[j,3]
                state_i = np.sort(state_i)
                if check(state_i, Nflav):
                    newstate.append(state_i)
    return np.array(newstate)

# Find all states a given initial state can transition into.
def stateFinder(instate, Nnu, Nflav, pkectrans):
    binom = np.array(list(combinations([i for i in range(Nnu)],2)))
    Nbinom = len(binom)
    nnewstate = 10  # > 1 to enter the while loop
    p_states = np.array([instate])
    newstate1 = np.array([np.zeros(Nnu), instate])
    trial = 0
    while nnewstate > 1:
        print(f'---------------- H^{trial+1} ----------------')
        newstate2 = np.zeros((1,Nnu))
        for i in range(1,len(newstate1)):
            newi = apply(newstate1[i], Nflav, Nbinom, binom, pkectrans)
            if len(newi) > 0:
                newstate2 = np.append(newstate2, newi, axis=0)
           
        newstate1 = np.array([np.zeros(Nnu)])
        for j in range(1,len(newstate2)):
            dist = np.sum(np.abs(p_states - newstate2[j]), axis=1)
            if np.min(dist) > 1e-7:
                p_states = np.append(p_states, [newstate2[j].astype(int)], axis=0)
                newstate1 = np.append(newstate1, [newstate2[j]], axis=0)
        nnewstate = len(newstate1)
        print(f'Number of new states(mod flavor choice) at this round: {nnewstate-1}')
        print(f'Number of states(mod flavor choice) visited so far: {len(p_states)}')
        trial += 1
    print(f'Number of momentum modes pair with conserved P and E is {len(p_states)}')
    # Calculate number of states
    Ns = 0
    for i in range(len(p_states)):
        #totp = np.sum(ps[p_states[i]], axis=0)
        #totke = np.sum(kes[p_states[i]])
        nstate = 2**Nnu/4**(Nnu-len(set(p_states[i])))
        Ns += nstate
        # Sanity Check
        #if np.sum(np.abs(totp-pinit)) > 1e-7 or np.abs(totke-keinit) > 1e-7:
        #    print('Total momentum or kinetic energy is not conserved!')
    print(f'Number of states with conserved P,E, and arbitrary flavor contents is {Ns:.0f}')
    bins_visited = np.sort(np.array(list(set(p_states.flatten()))))
    print(f'Activated momentum modes:')
    print(bins_visited)
    return p_states

# Given neutrino's mode index and flavor, return its bin number k = K*flavor + p
def bin(p, Nflav, flav):
    return Nflav*p + flav

# Given a state's binary representation, return its index representation
def b_to_j(b, Nbs):
    oc = []
    for i in range(Nbs):
        if b[i]==1:
            oc.append(i)
    bstr = ','.join(str(x) for x in oc)
    return bstr_to_j[bstr]

# Given a state's index representation, return its binary representation
def j_to_b(j, Nbs):
    bstr = j_to_bstr[j]
    oc = [int(x) for x in bstr.split(',')]
    b = [0]*Nbs
    for i in range(len(oc)):
        b[oc[i]] = 1
    return b

# Generate flavor informations.
def flavInfo(Nflav):
    print("Generating neutrino flavor pairs...")
    flavPairs = []
    for i in range(Nflav):
        for j in range(Nflav):
            flavPairs.append([i,j])
    flavPairs = np.array(flavPairs)
    return flavPairs

# Applying a*(b1)a(b2) to a basis state. Note b = [b1,b2]
def quad(b, basis):
    basis_copy = basis.copy()
    truth = True
    f = 1.0
    if basis_copy[b[1]] == 0:
        truth = False
    else:
        basis_copy[b[1]] = 0
        f = f * (-1)**np.sum(basis_copy[:b[1]])
    if basis_copy[b[0]] ==1:
        truth = False
    else:
        basis_copy[b[0]] = 1
        f = f * (-1)**np.sum(basis_copy[:b[0]])
    return truth, f, basis_copy


# Applying a*(b1)a*(b2)a(b3)a(b4) to state. Note b = [b1,b2,b3,b4]
def quar(b, basis):
    basis_copy = basis.copy()
    truth = True
    f = 1.0
    if basis_copy[b[3]] == 0:
        truth = False
    else:
        basis_copy[b[3]] = 0
        f = f * (-1)**np.sum(basis_copy[:b[3]])
    if basis_copy[b[2]] == 0:
        truth = False
    else:
        basis_copy[b[2]] = 0
        f = f * (-1)**np.sum(basis_copy[:b[2]])
    if basis_copy[b[1]] ==1:
        truth = False
    else:
        basis_copy[b[1]] = 1
        f = f * (-1)**np.sum(basis_copy[:b[1]])
    if basis_copy[b[0]] ==1:
        truth = False
    else:
        basis_copy[b[0]] = 1
        f = f * (-1)**np.sum(basis_copy[:b[0]])
    return truth, f, basis_copy

# Generate mass term given basis state j
def mass(j, Ns, Nps, Nflav, Pstates, flavPairs):
    instate = j_to_b(j, Nflav*Nps)
    state = np.zeros(Ns, dtype=complex)
    for p in range(Nps):
        kflavs = []
        for flav in range(Nflav):
            kflavs.append(bin(p,Nflav,flav))
        kflavs = np.array(kflavs)
        for pair in range(len(flavPairs)):
            
            # NEED TO DEFINE FACTORS STILL
            
            kf1 = flavPairs[pair,0]
            kf2 = flavPairs[pair,1]
            truth, fa, outstate = quad([kflavs[kf1],kflavs[kf2]], instate)
            if truth is True:
                state[b_to_j(outstate, Nflav*Nps)] += fa #* massfactors[pair]
    return state


# Generate full two body interaction term
def vvFull(j, Ns, Nps, Nflav, momenta4, flavPairs):
    instate = j_to_b(j, Nflav*Nps)
    state = np.zeros(Ns, dtype=complex)
    for i in range(len(momenta4)):
        p1 = momenta4[i,0]
        p2 = momenta4[i,1]
        q1 = momenta4[i,2]
        q2 = momenta4[i,3]
        factor = - gfs[i]
        for flav in range(len(flavPairs)):
            ip1 = bin(p1, flavPairs[flav,0])
            ip2 = bin(p1, flavPairs[flav,1])
            iq1 = bin(p1, flavPairs[flav,0])
            iq2 = bin(p1, flavPairs[flav,1])
            truth, fa, outstate = quar([ip1,ip2,iq1,iq2], instate)
            if truth is True:
                state[b_to_j(outstate, Nflav*Nps)] += fa*factor
    return state

# Construct the Hamiltonian.
def buildH(Ns, Nps, Nflav, Pstates, pkectrans, momenta4, gfs):
    H = np.zeros((Ns, Ns), dtype=complex)
    flavPairs = flavInfo(Nflav)
    for i in range(Ns):
        if i%100 == 0:
            print(f"Generating {i}th column of the Hamiltonian..."
        H[:,i] += mass(i, Ns, Nps, Nflav, Pstates, flavPairs)
        H[:,i] += vvFull(i, Ns, Nps, Nflav, momenta4, flavPairs)
    print("The Hamiltonian has be generated.")
    return H

# 
def main():
    return

#if __name__ == '__main__':
#    Move excess code into __name__ gaurd
#    main()

# Formatting
np.set_printoptions(formatter={'all': lambda x: "{:.12g}".format(x)})

# 
try:
    Nnu = int(input("Enter the number of neutrinos: "))     # Number of Neutrinos
    Nflav = int(input("Enter the number of flavors: "))     # Number of Flavor states
    zmax = int(input("Enter the momentum threshold: "))     # Used for generating Momentum modes
except ValueError:
    print("Invalid Input")
except Exception as expt:
    print(f"An unexpected error occurred: {expt}")

Pstates, Nps, pkectrans, momenta4, gfs = pGenerator(zmax)

Nbs = Nflav*Nps



