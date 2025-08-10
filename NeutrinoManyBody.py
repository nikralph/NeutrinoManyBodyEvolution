# This program is based on the paper https://arxiv.org/pdf/2404.16690
# Thank you Vincenzo Cirigliano and Yukari Yamauchi!

import numpy as np
import matplotlib.pyplot as plt
#import sys

from itertools import combinations
from collections import Counter
from math import comb

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
    for y in zyrange:
        for x in zxrange:
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
    for i in range(len(state)-Nflav):
        if np.var(state[i:i+1+Nflav]) < 1e-7:
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
def stateFinder(instate, Nnu, Nflav, flavcons, pkectrans):
    binom = np.array(list(combinations([i for i in range(Nnu)], 2)))
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
    skipped = 0
    for i in range(len(p_states)):
        pscounter = Counter(p_states[i])
        Nrptemp = {}
        for p in sorted(pscounter.values()):
            Nrptemp[p] = Nrptemp.get(p, 0) + 1
        Nrp = [0 for x in range(max(Nrptemp.keys()))]
        for key in Nrptemp.keys():
            Nrp[key-1] = Nrptemp[key]
        #if all(Nnu - count*Nrp[count] >= 0 for count in range(len(Nrp))):
        
    print(f'Number of states with conserved P,E, and arbitrary flavor contents is {Ns:.0f}')
    bins_visited = np.sort(np.array(list(set(p_states.flatten()))))
    print(f'Activated momentum modes:')
    print(bins_visited)
    # Find all states with arbitrary flavor contents.
    
    # NEED TO GENERALIZE BELOW BLOCK STILL FOR NFLAV >= 3
    
    # TRY: rp = [nrp, rp1, rp2, ...] and use (index = number of repeated modes) for checking each modes-Nflavs mod Nflav
    # Check flavor conservation when using only Hvv term.
    states = []
    if Nflav == 1:
        for l in p_states:
            #nstate = int(2**len(l)/4**(len(l)-len(list(set(l)))))
            nstate = int(Nflav**len(l)/(Nflav**Nflav)**(len(l)-len(list(set(l)))))
            rp = []
            #if nstate < 2**len(l):
            if nstate < Nflav**len(l):
                for i in range(len(l)-1):
                    if l[i] == l[i+1]:
                        rp.append(l[i])
            nrp = [x for x in l if x not in rp]
            Nnrp = len(nrp)
            for i in range(nstate):
                state = []
                for j in range(len(rp)):
                    for flav in Nflav:
                        state.append(Nflav*rp[j]+flav)
                    #state.append(2*rp[j])
                    #state.append(2*rp[j]+1)
                for j in range(Nnrp):
                    #state.append(2*nrp[j]+(i//2**(Nnrp-j-1))%2)
                    state.append(Nflav*nrp[j]+(i//Nflav**(Nnrp-j-1))%Nflav)
                states.append(sorted(state))
    elif Nflav == 2:
        Ne = flavcons[0]
        for l in p_states:
            nstate = int(2**len(l)/4**(len(l)-len(list(set(l))))) # all possible states (no flavor dependence)
            Nrp = len(l)-len(set(l))                              # number of repeats
            fc_nstate = comb(len(l)-2*Nrp, Ne-Nrp)
            print(f"nstate: {fc_nstate}, l={l}")
            
            rp = []                         #repeated momentum modes
            if nstate < 2**len(l):
                for i in range(len(l)-1):
                    if l[i]==l[i+1]:
                        rp.append(l[i])

            nrp = [x for x in l if x not in rp] #nonrepeated momentum modes
            numnrp = len(nrp)                   #number of such modes

            #parity of integer in state represents flavor; 2^len(l) flavor combinations per l for nonrepeated
            for i in range(nstate):
                
                state = []
                evens = 0                   # number of even integers in state (number of electron neutrinos)

                for j in range(len(rp)):        # append duplicated modes twice
                    state.append(2*rp[j])
                    state.append(2*rp[j]+1)
                    evens+=1
                
                for j in range(numnrp):
                    value = (2*nrp[j]+(i//2**(numnrp-j-1))%2) # counts in binary
                    state.append(value)
                    
                    if value % 2 == 0:
                           evens += 1

                if evens == Ne:
                    states.append(sorted(state))
    Ns = len(states)
    print(f"Ns: {Ns}")
    bstr_to_j = {}
    j_to_bstr = {}
    for i in range(Ns):
        b = ','.join(str(int(x)) for x in states[i])
        bstr_to_j[b] = i
        j_to_bstr[i] = b
    return Ns, p_states, bstr_to_j, j_to_bstr

# Given neutrino's mode index and flavor, return its bin number k = K*flavor + p
def bin(p, Nflav, flav):
    return Nflav*p + flav

# Given a state's binary representation, return its index representation
def b_to_j(b, Nbs, bstr_to_j):
    oc = []
    for i in range(Nbs):
        if b[i]==1:
            oc.append(i)
    bstr = ','.join(str(x) for x in oc)
    return bstr_to_j[bstr]

# Given a state's index representation, return its binary representation
def j_to_b(j, Nbs, j_to_bstr):
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
def mass(j, Ns, Nps, Nflav, Nbs, Pstates, flavPairs, bstr_to_j, j_to_bstr, tbar, wbar, angle):
    instate = j_to_b(j, Nbs, j_to_bstr)
    state = np.zeros(Ns, dtype=complex)
    for p in range(Nps):
        kflavs = []
        for flav in range(Nflav):
            kflavs.append(bin(p, Nflav, flav))
        kflavs = np.array(kflavs)
        
        # NEED TO DEFINE FACTORS STILL
        absp = np.linalg.norm(Pstates[p])
        factor_ee = tbar*absp - np.cos(2*angle)*wbar/absp
        factor_mm = tbar*absp + np.cos(2*angle)*wbar/absp
        factor_em = np.sin(2*angle)*wbar/absp
        factor_me = factor_em
        massfactors = np.array([factor_ee, factor_em, factor_me, factor_mm], dtype=complex)
        
        for pair in range(len(flavPairs)):
            kf1 = flavPairs[pair,0]
            kf2 = flavPairs[pair,1]
            truth, fa, outstate = quad([kflavs[kf1],kflavs[kf2]], instate)
            if truth is True:
                state[b_to_j(outstate, Nbs, bstr_to_j)] += fa * massfactors[pair]
    return state

# Generate full two body interaction term
def vvFull(j, Ns, Nps, Nflav, Nbs, momenta4, gfs, flavPairs, bstr_to_j, j_to_bstr):
    instate = j_to_b(j, Nbs, j_to_bstr)
    state = np.zeros(Ns, dtype=complex)
    for i in range(len(momenta4)):
        p1 = momenta4[i,0]
        p2 = momenta4[i,1]
        q1 = momenta4[i,2]
        q2 = momenta4[i,3]
        factor = - gfs[i]
        for flav in range(len(flavPairs)):
            ip1 = bin(p1, Nflav, flavPairs[flav,0])
            ip2 = bin(p2, Nflav, flavPairs[flav,1])
            iq1 = bin(q1, Nflav, flavPairs[flav,0])
            iq2 = bin(q2, Nflav, flavPairs[flav,1])
            truth, fa, outstate = quar([ip1,ip2,iq1,iq2], instate)
            if truth is True:
                state[b_to_j(outstate, Nbs, bstr_to_j)] += fa * factor
    return state

# Construct the Hamiltonian.
def buildH(Ns, Nps, Nflav, Nbs, Pstates, pkectrans, momenta4, gfs, bstr_to_j, j_to_bstr, tbar, wbar, angle):
    H = np.zeros((Ns, Ns), dtype=complex)
    flavPairs = flavInfo(Nflav)
    for i in range(Ns):
        if i%100 == 0:
            print(f"Generating {i}th column of the Hamiltonian...")
        # Ignore Mass term for now; In dense media we expact interactions to supress vaccume oscillations, so the Hvv term should dominate. Test with mass term later.
        #H[:,i] += mass(i, Ns, Nps, Nflav, Nbs, Pstates, flavPairs, bstr_to_j, j_to_bstr, tbar, wbar, angle)
        H[:,i] += vvFull(i, Ns, Nps, Nflav, Nbs, momenta4, gfs, flavPairs, bstr_to_j, j_to_bstr)
    print("The Hamiltonian has be generated.")
    return H

# Returns observables
def observable(Nbs, Ns, Nflav, state, j_to_bstr):
    obs = np.zeros(Nbs)
    for i in range(Ns):
        binary = j_to_b(i, Nbs, j_to_bstr)
        obs += np.abs(state[i]**2*np.array(binary))
    return obs

# returns string of time and wave function amplitude
def print_cstr(state, i, dt):
    return str(i*dt) + ' ' + ' '.join([str(x) for x in state]) 

# returns string of time and occupation number per bin
def print_nstr(state, i, Nbs, Ns, Nflav, dt, j_to_bstr):
    obs = observable(Nbs, Ns, Nflav, state, j_to_bstr)
    obslist = [x for x in obs]
    return str(i*dt) + ' ' + ' '.join([str(x) for x in obs]), obslist

# Microcononical Ensenbal N+ Observable
def mcNplus(Ns, Nbs, Nps, Nnu, Nflav, flavcons, j_to_bstr):
    mcstate = (1/np.sqrt(Ns))*np.ones(Ns,dtype=complex)
    mcObs = observable(Nbs, Ns, Nflav, mcstate, j_to_bstr)
    plt.rcParams['text.usetex'] = True
    graph = plt.figure(figsize=(19.2,10.8))
    x = np.linspace(0, Nps, Nps)
    mcNpObs = np.zeros((1,Nps))
    for i in range(Nps):
        for flav in range(Nflav):
            mcNpObs[0,i] += mcObs[Nflav*i+flav]
    plt.scatter(x,mcNpObs[0,:])
    plt.title(r'Microcanonical ensemble: $N_{i}^{+}N$ Observable ' + f'for Nflav = {Nflav}')
    plt.xlim([0-1,Nps+1])
    plt.ylim([-0.05,1.05])
    plt.grid()
    plt.xlabel(r'time $(\varepsilon^{-1})$')
    plt.ylabel(r'$N_{i}^{+}N$')
    plt.savefig(f'MC_Nflav{Nflav}_Np{Nps}_Nnu{Nnu}_Ne{flavcons[0]}.png')
    plt.close(graph)
    return

# Graph generator for momentum observable
def nplus(instate, U, Nbs, Ns, Nnu, Nps, Nflav, flavcons, dt, Nt, j_to_bstr):
    state = instate.copy()
    print("Generating graph for momentum observable...")
    plt.rcParams['text.usetex'] = True
    graph = plt.figure(figsize=(19.2,10.8))
    times = np.linspace(0, dt*Nt, Nt+1)
    #npobs = np.array([])
    obsstr, obslist = print_nstr(state, 0, Nbs, Ns, Nflav, dt, j_to_bstr)
    obs = np.zeros((1,Nps))
    for j in range(Nps):
        for flav in range(Nflav):
            obs[0,j] += (1/Nflav)*obslist[Nflav*j+flav]
    npobs = obs.T.copy()
    for i in range(1, Nt+1):
        state = U @ state
        n = np.linalg.norm(state)
        if abs(n-1.0) > 1e-5: # Sanity Check
            print('Norm off by > 1e-5 at time ', i*dt)
        obs = np.zeros((1,Nps))
        obsstr, obslist = print_nstr(state, i, Nbs, Ns, Nflav, dt, j_to_bstr)
        for j in range(Nps):
            for flav in range(Nflav):
                obs[0,j] += obslist[Nflav*j+flav]
        npobs = np.concatenate((npobs, obs.T), axis=1)
    
    # Sanity Checks
    print(f'Size of times: {len(times)}')
    print(f'Size of npobs: {len(npobs)}')
    print(f'Size of one observable list: {len(npobs[0])}')
    print(f'Initial state: ' + str(npobs[:,0]))
    
    for i in range(Nps):
        plt.plot(times, npobs[i], color='black', linewidth=0.5)
    plt.title(r'$N_{i}^{+}N$ Observable ' + f'for Nflav = {Nflav}')
    plt.xlim([0,Nt*dt])
    plt.xlabel(r'time $(\varepsilon^{-1})$')
    plt.ylabel(r'$N_{i}^{+}N$')
    plt.savefig(f'Nflav{Nflav}_Np{Nps}_Nnu{Nnu}_time{int(dt*Nt)}_Ne{flavcons[0]}.png')
    plt.close(graph)
    return

# Graph generator for flavor observable
def nminus(instate, U, Nbs, Ns, Nnu, Nps, Nflav, dt, Nt, j_to_bstr):
    state = instate.copy()
    print("Generating graph for N+ observable...")
    plt.rcParams['text.usetex'] = True
    graph = plt.figure(figsize=(8,6))
    times = np.linspace(0, dt*Nt, Nt+1)
    obsstr, obslist = print_nstr(state, 0, Nbs, Ns, Nflav, dt, j_to_bstr)
    obs = np.zeros(Nps)
    
    # TODO: Finish Generalizing Flavor Observable
    
    for i in range(Nps):
        plt.plot(times, nmobs[i], color='black', linewidth=0.5)
    plt.title(r'$N_{i}^{-}N$ Observable ' + f'for Nflav = {Nflav}')
    plt.xlim([0,Nt*dt])
    plt.xlabel(r'time $(\epsilon^{-1})$')
    plt.ylabel(r'$N_{i}^{-}N$')
    plt.show()
    return

# Perform desired simulations, while adjusting initial flavors to conserve.
def main():
    Nflav = 2
    zmax = 5
    instate = np.sort(np.array([0,5,8,10,12,20,25,26,28,33]))
    Nnu = len(instate)
    for Nmu in range(int(Nnu/2)):
        Ne = Nnu - Nmu
        print(f"Running simulation with Ne:{Ne} and Nmu:{Nmu}")
        flavcons = [Ne,Nmu]
        test(Nflav, flavcons, zmax, instate)
    return

# QA testing
def test(Nflav, flavcons, zmax, instate):
    Nnu = len(instate)
    
    # Sanity Checks
    if not (len(flavcons) == Nflav):
        raise ValueError("Length of flavcons must match Nflav.")
    if sum(flavcons) != len(instate):
        raise ValueError("Total flavor repetitions must exactly match the number of allowed Nps spots.")

    Pstates, Nps, pkectrans, momenta4, gfs = pGenerator(zmax)
    Ns, p_states, bstr_to_j, j_to_bstr = stateFinder(instate, Nnu, Nflav, flavcons, pkectrans)
    
    Nbs = Nflav*Nps
    tbar = 10**(4)
    wbar = 1
    angle = (1/2)*np.arcsin(0.8)
    dt = 0.01
    Nt = 1000
    
    H = buildH(Ns, Nps, Nflav, Nbs, Pstates, pkectrans, momenta4, gfs, bstr_to_j, j_to_bstr, tbar, wbar, angle)
    print("Diagonalizing the hamiltonian...")
    Evals, Evecs = np.linalg.eigh(H)
    print("Done diagonalizing hamiltonian.")
    U = Evecs @ np.diag(np.exp(-dt*Evals*1j)) @ Evecs.conj().T
    
    state = np.zeros(Ns, dtype=complex)
    
    # Starting with lower index as Ne, for simplicity.
    # It is worth investigate how our initial conditions effect our thermalization process.
    b = np.zeros(Nbs)
    for flav in range(Nflav):
        for Nf in range(flavcons[flav]):
            b[Nflav*instate[Nf+sum(flavcons[:flav])] + flav] = 1
    
    initj = b_to_j(b, Nbs, bstr_to_j)
    state[initj] = 1.0
    
    mcNplus(Ns, Nbs, Nps, Nnu, Nflav, flavcons, j_to_bstr)
    nplus(state, U, Nbs, Ns, Nnu, Nps, Nflav, flavcons, dt, Nt, j_to_bstr)
    return

def test1():
    zmax = 5
    Nflav = 1
    instate = np.sort(np.array([0,5,8,10,12,20,25,26,28,33]))
    Nnu = len(instate)
    flavcons = [Nnu]
    Pstates, Nps, pkectrans, momenta4, gfs = pGenerator(zmax)
    Ns, p_states, bstr_to_j, j_to_bstr = stateFinder(instate, Nnu, Nflav, flavcons, pkectrans)
    
    Nbs = Nflav*Nps
    tbar = 10**(4)
    wbar = 1
    angle = (1/2)*np.arcsin(0.8)
    dt = 0.01
    Nt = 1000
    
    H = buildH(Ns, Nps, Nflav, Nbs, Pstates, pkectrans, momenta4, gfs, bstr_to_j, j_to_bstr, tbar, wbar, angle)
    print("Diagonalizing the hamiltonian...")
    Evals, Evecs = np.linalg.eigh(H)
    print("Done diagonalizing hamiltonian.")
    U = Evecs @ np.diag(np.exp(-dt*Evals*1j)) @ Evecs.conj().T
    
    state = np.zeros(Ns, dtype=complex)
    
    b = np.zeros(Nbs, dtype=int)
    for k in instate:
        b[Nflav*k] = 1  # All same flavor
    
    trying = False
    while trying:
        bchoose = input(f"Choose initial state by inputting {Nnu} allowed momenta and flavors as a pair: momenta,flavor (use space to separate pairs): ")
        if len([x for x in bchoose.split(' ')]) == Nnu:
            trying = False
        b = np.zeros(Nbs, dtype=int)
        for things in bchoose.split(' '):
            i, j = [int(x) for x in things.split(',')]
            b[Nflav*i+j] = 1
            if j not in [0,1]:
                trying = True
            if Nflav*i > Nbs:
                trying = True
        #b = ' '.join([str(x) for x in b])
    initj = b_to_j(b, Nbs, bstr_to_j)
    state[initj] = 1.0

    nplus(state, U, Nbs, Ns, Nnu, Nps, Nflav, flavcons, dt, Nt, j_to_bstr)
    return

if __name__ == '__main__':
    main()
    #test1()

# Formatting
#np.set_printoptions(formatter={'all': lambda x: "{:.12g}".format(x)})
