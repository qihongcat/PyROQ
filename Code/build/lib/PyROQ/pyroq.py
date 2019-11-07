import numpy
import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import lal
import lalsimulation
from lal.lal import PC_SI as LAL_PC_SI
import h5py
import warnings
import random

# Calculating the projection of complex vector v on complex vector u
def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * numpy.vdot(v,u) / numpy.vdot(u,u) 

# Calculating the normalized residual (= a new basis) of a vector vec from known bases
def gram_schmidt(bases, vec):
    for i in numpy.arange(0,len(bases)):
        vec = vec - proj(bases[i], vec)
    return vec/numpy.sqrt(numpy.vdot(vec,vec)) # normalized new basis

# Calculating overlap of two waveforms
def overlap_of_two_waveforms(wf1, wf2):
    wf1norm = wf1/numpy.sqrt(numpy.vdot(wf1,wf1)) # normalize the first waveform
    wf2norm = wf2/numpy.sqrt(numpy.vdot(wf2,wf2)) # normalize the second waveform
    diff = wf1norm - wf2norm
    #overlap = 1 - 0.5*(numpy.vdot(diff,diff))
    overlap = numpy.real(numpy.vdot(wf1norm, wf2norm))
    return overlap

def spherical_to_cartesian(sph):
    x = sph[0]*numpy.sin(sph[1])*numpy.cos(sph[2])
    y = sph[0]*numpy.sin(sph[1])*numpy.sin(sph[2])
    z = sph[0]*numpy.cos(sph[1])
    car = [x,y,z]
    return car

def get_m1m2_from_mcq(mc, q):
    m2 = mc * q ** (-0.6) * (1+q)**0.2
    m1 = m2 * q
    return numpy.array([m1,m2])

def generate_a_waveform(m1, m2, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, distance, deltaF, f_min, f_max, waveFlags, approximant):
    test_mass1 = m1 * lal.lal.MSUN_SI
    test_mass2 = m2 * lal.lal.MSUN_SI
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2)     
    [plus_test, cross_test]=lalsimulation.SimInspiralChooseFDWaveform(test_mass1, test_mass2, spin1[0], spin1[1], spin1[2], spin2[0], spin2[1], spin2[2], distance, iota, phiRef, 0, ecc, 0, deltaF, f_min, f_max, 0, waveFlags, approximant)
    hp = plus_test.data.data
    hp_test = hp[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)]
    return hp_test

def generate_a_waveform_from_mcq(mc, q, spin1, spin2, ecc, lambda1, lambda2, iota, phiRef, distance, deltaF, f_min, f_max, waveFlags, approximant):
    m1,m2 = get_m1m2_from_mcq(mc,q)
    test_mass1 = m1 * lal.lal.MSUN_SI
    test_mass2 = m2 * lal.lal.MSUN_SI
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2) 
    [plus_test, cross_test]=lalsimulation.SimInspiralChooseFDWaveform(test_mass1, test_mass2, spin1[0], spin1[1], spin1[2], spin2[0], spin2[1], spin2[2], distance, iota, phiRef, 0, ecc, 0, deltaF, f_min, f_max, 0, waveFlags, approximant)
    hp = plus_test.data.data
    hp_test = hp[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)]
    return hp_test

def generate_params_points(npts, nparams, params_low, params_high):
    paramspoints = numpy.random.uniform(params_low, params_high, size=(npts,nparams))
    paramspoints = paramspoints.round(decimals=6)
    return paramspoints

# now generating N=npts waveforms at points that are 
# randomly uniformly distributed in parameter space
# and calculate their inner products with the 1st waveform
# so as to find the best waveform as the new basis
def least_match_waveform_unnormalized(paramspoints, known_bases, npts, distance, deltaF, f_min, f_max, waveFlags, approximant):
    overlaps = numpy.zeros(npts)
    modula = numpy.zeros(npts)
    for i in numpy.arange(0,len(paramspoints)):
        paramspoint = paramspoints[i]
        m1, m2 = get_m1m2_from_mcq(paramspoint[0],paramspoint[1])
        s1x, s1y, s1z = spherical_to_cartesian(paramspoint[2:5]) 
        s2x, s2y, s2z = spherical_to_cartesian(paramspoint[5:8]) 
        iota = paramspoint[8]  
        phiRef = paramspoint[9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoint[10]
        if len(paramspoint)==12:
            lambda1 = paramspoint[10]
            lambda2 = paramspoint[11]
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2) 
        f_ref = 0 
        RA=0    
        DEC=0   
        psi=0   
        phi=0   
        m1 *= lal.lal.MSUN_SI
        m2 *= lal.lal.MSUN_SI
        [plus,cross]=lalsimulation.SimInspiralChooseFDWaveform(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, distance, iota, phiRef, 0, 0, 0, deltaF, f_min, f_max, f_ref, waveFlags, approximant)
        hp_tmp = plus.data.data[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)] # data_tmp is hplus and is a complex vector 
        residual = hp_tmp
        for k in numpy.arange(0,len(known_bases)):
            residual -= proj(known_bases[k],hp_tmp)
        modula[i] = numpy.sqrt(numpy.vdot(residual, residual))
    arg_newbasis = numpy.argmax(modula)    
    mass1, mass2 = get_m1m2_from_mcq(paramspoints[arg_newbasis][0],paramspoints[arg_newbasis][1])
    mass1 *= lal.lal.MSUN_SI
    mass2 *= lal.lal.MSUN_SI
    sp1x, sp1y, sp1z = spherical_to_cartesian(paramspoints[arg_newbasis,2:5]) 
    sp2x, sp2y, sp2z = spherical_to_cartesian(paramspoints[arg_newbasis,5:8]) 
    inclination = paramspoints[arg_newbasis][8]
    phi_ref = paramspoints[arg_newbasis][9]
    ecc = 0
    if len(paramspoint)==11:
        ecc = paramspoints[arg_newbasis][10]
    if len(paramspoint)==12:
        lambda1 = paramspoints[arg_newbasis][10]
        lambda2 = paramspoints[arg_newbasis][11]
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2)     
    [plus_new, cross_new]=lalsimulation.SimInspiralChooseFDWaveform(mass1, mass2, sp1x, sp1y, sp1z, sp2x, sp2y, sp2z, distance, inclination, phi_ref, 0, ecc, 0, deltaF, f_min, f_max, 0, waveFlags, approximant)
    hp_new = plus_new.data.data
    hp_new = hp_new[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)]
    basis_new = gram_schmidt(known_bases, hp_new)
    return numpy.array([basis_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod


def least_match_quadratic_waveform_unnormalized(paramspoints, known_quad_bases, npts, distance, deltaF, f_min, f_max, waveFlags, approximant):
    overlaps = numpy.zeros(npts)
    modula = numpy.zeros(npts)
    for i in numpy.arange(0,len(paramspoints)):
        paramspoint = paramspoints[i]
        m1, m2 = get_m1m2_from_mcq(paramspoint[0],paramspoint[1])
        s1x, s1y, s1z = spherical_to_cartesian(paramspoint[2:5]) 
        s2x, s2y, s2z = spherical_to_cartesian(paramspoint[5:8]) 
        iota=paramspoint[8]  
        phiRef=paramspoint[9]
        ecc = 0
        if len(paramspoint)==11:
            ecc = paramspoint[10]
        if len(paramspoint)==12:
            lambda1 = paramspoint[10]
            lambda2 = paramspoint[11]
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
            lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2) 
        f_ref = 0 
        RA=0    
        DEC=0   
        psi=0   
        phi=0   
        m1 *= lal.lal.MSUN_SI
        m2 *= lal.lal.MSUN_SI
        [plus,cross]=lalsimulation.SimInspiralChooseFDWaveform(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, distance, iota, phiRef, 0, 0, 0, deltaF, f_min, f_max, f_ref, waveFlags, approximant)
        hp_tmp = plus.data.data[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)] # data_tmp is hplus and is a complex vector 
        hp_quad_tmp = (numpy.absolute(hp_tmp))**2
        residual = hp_quad_tmp
        for k in numpy.arange(0,len(known_quad_bases)):
            residual -= proj(known_quad_bases[k],hp_quad_tmp)
        modula[i] = numpy.sqrt(numpy.vdot(residual, residual))
    arg_newbasis = numpy.argmax(modula)    
    mass1, mass2 = get_m1m2_from_mcq(paramspoints[arg_newbasis][0],paramspoints[arg_newbasis][1])
    mass1 *= lal.lal.MSUN_SI
    mass2 *= lal.lal.MSUN_SI
    sp1x, sp1y, sp1z = spherical_to_cartesian(paramspoints[arg_newbasis,2:5]) 
    sp2x, sp2y, sp2z = spherical_to_cartesian(paramspoints[arg_newbasis,5:8]) 
    inclination = paramspoints[arg_newbasis][8]
    phi_ref = paramspoints[arg_newbasis][9]
    ecc = 0
    if len(paramspoint)==11:
        ecc = paramspoints[arg_newbasis][10]
    if len(paramspoint)==12:
        lambda1 = paramspoints[arg_newbasis][10]
        lambda2 = paramspoints[arg_newbasis][11]
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(waveFlags, lambda1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(waveFlags, lambda2)     
    [plus_new, cross_new]=lalsimulation.SimInspiralChooseFDWaveform(mass1, mass2, sp1x, sp1y, sp1z, sp2x, sp2y, sp2z, distance, inclination, phi_ref, 0, ecc, 0, deltaF, f_min, f_max, 0, waveFlags, approximant)
    hp_new = plus_new.data.data
    hp_new = hp_new[numpy.int(f_min/deltaF):numpy.int(f_max/deltaF)]
    hp_quad_new = (numpy.absolute(hp_new))**2
    basis_quad_new = gram_schmidt(known_quad_bases, hp_quad_new)    
    return numpy.array([basis_quad_new, paramspoints[arg_newbasis], modula[arg_newbasis]]) # elements, masses&spins, residual mod

def bases_searching_results_unnormalized(npts, nparams, nbases, known_bases, basis_waveforms, params, residual_modula, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    if nparams == 10: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), iota, and phiRef\n")
    if nparams == 11: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), eccentricity, iota, and phiRef\n")
    if nparams == 12: print("The parameters are Mc, q, s1(mag, theta, phi), s2(mag, theta, phi), lambda1, lambda2, iota, and phiRef\n") 
    for k in numpy.arange(0,nbases-1):
        params_points = generate_params_points(npts, nparams, params_low, params_high)
        basis_new, params_new, rm_new= least_match_waveform_unnormalized(params_points, known_bases, npts, distance, deltaF, f_min, f_max, waveFlags, approximant)
        print("Linear Iter: ", k, params_new)
        known_bases= numpy.append(known_bases, numpy.array([basis_new]), axis=0)
        params = numpy.append(params, numpy.array([params_new]), axis = 0)
        residual_modula = numpy.append(residual_modula, rm_new)
    return known_bases, params, residual_modula

def bases_searching_quadratic_results_unnormalized(npts, nparams, nbases_quad, known_quad_bases, basis_waveforms, params_quad, residual_modula, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    for k in numpy.arange(0,nbases_quad-1):
        print("Quadratic Iter: ", k)
        params_points = generate_params_points(npts, nparams, params_low, params_high)
        basis_new, params_new, rm_new= least_match_quadratic_waveform_unnormalized(params_points, known_quad_bases, npts, distance, deltaF, f_min, f_max, waveFlags, approximant)
        known_quad_bases= numpy.append(known_quad_bases, numpy.array([basis_new]), axis=0)
        params_quad = numpy.append(params_quad, numpy.array([params_new]), axis = 0)
        residual_modula = numpy.append(residual_modula, rm_new)
    return known_quad_bases, params_quad, residual_modula

def massrange(mc_low, mc_high, q_low, q_high):
    mmin = get_m1m2_from_mcq(mc_low,q_high)[1]
    mmax = get_m1m2_from_mcq(mc_high,q_high)[0]
    return [mmin, mmax]

def initial_basis(mc_low, mc_high, q_low, q_high, s1sphere_low, s1sphere_high, s2sphere_low, s2sphere_high, ecc_low, ecc_high, lambda1_low, lambda1_high, lambda2_low, lambda2_high, iota_low, iota_high, phiref_low, phiref_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    try:
        if approximant==lalsimulation.IMRPhenomPv2:
            nparams = 10
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, 0, 0, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant)
    except AttributeError: 
        pass
    try:
        if approximant==lalsimulation.IMRPhenomPv3:
            nparams = 10
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, 0, 0, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant)
    except AttributeError: 
        pass
    try:
        if approximant==lalsimulation.IMRPhenomPv3HM:
            nparams = 10
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, 0, 0, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant)
    except AttributeError: 
        pass
    try:
        if approximant==lalsimulation.IMRPhenomXHM:
            nparams = 10
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, 0, 0, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant)
    except AttributeError: 
        pass
    try:
        if approximant==lalsimulation.TaylorF2Ecc:
            nparams = 11
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], ecc_low, iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], ecc_low, iota_high, phiref_high]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], ecc_low, 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), ecc_low, 0, 0, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant)
    except AttributeError: 
        pass
    try:    
        if approximant==lalsimulation.IMRPhenomPv2_NRTidal:
            nparams = 12
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], lambda1_low, lambda2_low, iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], lambda1_high, lambda2_high, ]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], lambda1_low, lambda2_low, 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, lambda1_low, lambda2_low, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant) 
    except AttributeError: 
        pass
    try:
        if approximant==lalsimulation.IMRPhenomNSBH:
            nparams = 12
            params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], lambda1_low, lambda2_low, iota_low, phiref_low] 
            params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], lambda1_high, lambda2_high, ]
            params_start = numpy.array([[mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], lambda1_low, lambda2_low, 0.33333*np.pi, 1.5*np.pi]])
            hp1 = generate_a_waveform_from_mcq(mc_low, q_low, spherical_to_cartesian(s1sphere_low), spherical_to_cartesian(s2sphere_low), 0, lambda1_low, lambda2_low, iota_low, phiref_low, distance, deltaF, f_min, f_max, waveFlags, approximant) 
    except AttributeError: 
        pass
    return numpy.array([nparams, params_low, params_high, params_start, hp1])

def empnodes(ndim, known_bases): # Here known_bases is the full copy known_bases_copy. Its length is equal to or longer than ndim.
    emp_nodes = numpy.arange(0,ndim)*100000000
    emp_nodes[0] = numpy.argmax(known_bases[0])
    c1 = known_bases[1,emp_nodes[0]]/known_bases[0,1]
    interp1 = numpy.multiply(c1,known_bases[0])
    diff1 = interp1 - known_bases[1]
    r1 = numpy.absolute(diff1)
    emp_nodes[1] = numpy.argmax(r1)
    for k in numpy.arange(2,ndim):
        emp_tmp = emp_nodes[0:k]
        Vtmp = numpy.transpose(known_bases[0:k,emp_tmp])
        inverse_Vtmp = numpy.linalg.inv(Vtmp)
        e_to_interp = known_bases[k]
        Ci = numpy.dot(inverse_Vtmp, e_to_interp[emp_tmp])
        interpolantA = numpy.zeros(len(known_bases[k]))+numpy.zeros(len(known_bases[k]))*1j
        for j in numpy.arange(0, k):
            tmp = numpy.multiply(Ci[j], known_bases[j])
            interpolantA += tmp
        diff = interpolantA - known_bases[k]
        r = numpy.absolute(diff)
        emp_nodes[k] = numpy.argmax(r)
        emp_nodes = sorted(emp_nodes)
    u, c = numpy.unique(emp_nodes, return_counts=True)
    dup = u[c > 1]
    #print(len(emp_nodes), "\nDuplicates indices:", dup)
    emp_nodes = numpy.unique(emp_nodes)
    ndim = len(emp_nodes)
    #print(len(emp_nodes), "\n", emp_nodes)
    V = numpy.transpose(known_bases[0:ndim, emp_nodes])
    inverse_V = numpy.linalg.inv(V)
    return numpy.array([ndim, inverse_V, emp_nodes])

def surroerror(ndim, inverse_V, emp_nodes, known_bases, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant):
    hp_test = generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
    Ci = numpy.dot(inverse_V, hp_test[emp_nodes])
    interpolantA = numpy.zeros(len(hp_test))+numpy.zeros(len(hp_test))*1j
    #ndim = len(known_bases)
    for j in numpy.arange(0, ndim):
        tmp = numpy.multiply(Ci[j], known_bases[j])
        interpolantA += tmp
    surro = (1-overlap_of_two_waveforms(hp_test, interpolantA))*deltaF
    return surro

def surros(tolerance, ndim, inverse_V, emp_nodes, known_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant): # Here known_bases is known_bases_copy
    test_points = generate_params_points(nts, nparams, params_low, params_high)
    surros = numpy.zeros(nts)
    count = 0
    for i in numpy.arange(0,nts):
        test_mc =  test_points[i,0]
        test_q = test_points[i,1]
        test_s1 = spherical_to_cartesian(test_points[i,2:5])
        test_s2 = spherical_to_cartesian(test_points[i,5:8])
        test_iota = test_points[i,8]
        test_phiref = test_points[i,9]
        test_ecc = 0
        test_lambda1 = 0
        test_lambda2 = 0
        if nparams == 11: test_ecc = test_points[i,10]
        if nparams == 12: 
            test_lambda1 = test_points[i,10]
            test_lambda2 = test_points[i,11]
        surros[i] = surroerror(ndim, inverse_V, emp_nodes, known_bases[0:ndim], test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
        if (surros[i] > tolerance):
            count = count+1
    print(ndim, "basis elements gave", count, "bad points of surrogate error > ", tolerance)
    if count == 0: val =0
    else: val = 1
    return val

def roqs(tolerance, freq,  ndimlow, ndimhigh, ndimstepsize, known_bases_copy, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    for num in np.arange(ndimlow, ndimhigh, ndimstepsize):
        ndim, inverse_V, emp_nodes = empnodes(num, known_bases_copy)
        if surros(tolerance, ndim, inverse_V, emp_nodes, known_bases_copy, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)==0:
            b_linear = numpy.dot(numpy.transpose(known_bases_copy[0:ndim]),inverse_V)
            f_linear = freq[emp_nodes]
            numpy.save('./B_linear.npy',numpy.transpose(b_linear))
            numpy.save('./fnodes_linear.npy',f_linear)
            print("Number of linear basis elements is ", ndim, "and the linear ROQ data save in B_linear.npy")
            break
    return

def testrep(b_linear, emp_nodes, test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant):
    hp_test = generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
    hp_test_emp = hp_test[emp_nodes]
    hp_rep = numpy.dot(b_linear,hp_test_emp)
    diff = hp_rep - hp_test
    rep_error = diff/numpy.sqrt(numpy.vdot(hp_test,hp_test))
    plt.plot(numpy.real(rep_error), label='Real part of h+') 
    plt.plot(numpy.imag(rep_error), label='Imaginary part of h+')
    plt.xlabel('Waveform Node Number')
    plt.ylabel('Fractional Representation Error')
    plt.title('Rep Error with numpy.linalg.inv()')
    plt.legend(loc=0)
    plt.show()
    return

def empnodes_quad(ndim_quad, known_quad_bases):
    emp_nodes_quad = numpy.arange(0,ndim_quad)*100000000
    emp_nodes_quad[0] = numpy.argmax(known_quad_bases[0])
    c1_quad = known_quad_bases[1,emp_nodes_quad[0]]/known_quad_bases[0,1]
    interp1_quad = numpy.multiply(c1_quad,known_quad_bases[0])
    diff1_quad = interp1_quad - known_quad_bases[1]
    r1_quad = numpy.absolute(diff1_quad)
    emp_nodes_quad[1] = numpy.argmax(r1_quad)
    for k in numpy.arange(2,ndim_quad):
        emp_tmp_quad = emp_nodes_quad[0:k]
        Vtmp_quad = numpy.transpose(known_quad_bases[0:k,emp_tmp_quad])
        inverse_Vtmp_quad = numpy.linalg.inv(Vtmp_quad)
        e_to_interp_quad = known_quad_bases[k]
        Ci_quad = numpy.dot(inverse_Vtmp_quad, e_to_interp_quad[emp_tmp_quad])
        interpolantA_quad = numpy.zeros(len(known_quad_bases[k]))+numpy.zeros(len(known_quad_bases[k]))*1j
        for j in numpy.arange(0, k):
            tmp_quad = numpy.multiply(Ci_quad[j], known_quad_bases[j])
            interpolantA_quad += tmp_quad
        diff_quad = interpolantA_quad - known_quad_bases[k]
        r_quad = numpy.absolute(diff_quad)
        emp_nodes_quad[k] = numpy.argmax(r_quad)
        emp_nodes_quad = sorted(emp_nodes_quad)
    u_quad, c_quad = numpy.unique(emp_nodes_quad, return_counts=True)
    dup_quad = u_quad[c_quad > 1]
    #print(len(emp_nodes_quad), "\nduplicates quad indices:", dup_quad)
    emp_nodes_quad = numpy.unique(emp_nodes_quad)
    ndim_quad = len(emp_nodes_quad)
    #print(len(emp_nodes_quad), "\n", emp_nodes_quad)
    V_quad = numpy.transpose(known_quad_bases[0:ndim_quad,emp_nodes_quad])
    inverse_V_quad = numpy.linalg.inv(V_quad)
    return numpy.array([ndim_quad, inverse_V_quad, emp_nodes_quad])

def surroerror_quad(ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant):
    hp_test_quad = (numpy.absolute(generate_a_waveform_from_mcq(test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant)))**2
    Ci_quad = numpy.dot(inverse_V_quad, hp_test_quad[emp_nodes_quad])
    interpolantA_quad = numpy.zeros(len(hp_test_quad))+numpy.zeros(len(hp_test_quad))*1j    
    #ndim_quad = len(known_quad_bases)
    for j in numpy.arange(0, ndim_quad):
        tmp_quad = numpy.multiply(Ci_quad[j], known_quad_bases[j])
        interpolantA_quad += tmp_quad
    surro_quad = (1-overlap_of_two_waveforms(hp_test_quad, interpolantA_quad))*deltaF
    return surro_quad

def surros_quad(tolerance_quad, ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    test_points = generate_params_points(nts, nparams, params_low, params_high)
    surros = numpy.zeros(nts)
    count = 0
    for i in numpy.arange(0,nts):
        test_mc_quad =  test_points[i,0]
        test_q_quad = test_points[i,1]
        test_s1_quad = spherical_to_cartesian(test_points[i,2:5])
        test_s2_quad = spherical_to_cartesian(test_points[i,5:8])
        test_iota_quad = test_points[i,8]
        test_phiref_quad = test_points[i,9]
        test_ecc_quad = 0
        test_lambda1_quad = 0
        test_lambda2_quad = 0
        if nparams == 11: test_ecc_quad = test_points[i,10]
        if nparams == 12: 
            test_lambda1_quad = test_points[i,10]
            test_lambda2_quad = test_points[i,11]
        surros[i] = surroerror_quad(ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases[0:ndim_quad], test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant)
        if (surros[i] > tolerance_quad):
            count = count+1
    print(ndim_quad, "basis elements gave", count, "bad points of surrogate error > ", tolerance_quad)
    if count == 0: val =0
    else: val = 1
    return val

def roqs_quad(tolerance_quad, freq,  ndimlow_quad, ndimhigh_quad, ndimstepsize_quad, known_quad_bases_copy, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant):
    for num in np.arange(ndimlow_quad, ndimhigh_quad, ndimstepsize_quad):
        ndim_quad, inverse_V_quad, emp_nodes_quad = empnodes_quad(num, known_quad_bases_copy)
        if surros_quad(tolerance_quad, ndim_quad, inverse_V_quad, emp_nodes_quad, known_quad_bases_copy, nts, nparams, params_low, params_high, distance, deltaF, f_min, f_max, waveFlags, approximant)==0:
            b_quad = numpy.dot(numpy.transpose(known_quad_bases_copy[0:ndim_quad]), inverse_V_quad)
            f_quad = freq[emp_nodes_quad]
            numpy.save('./B_quadratic.npy', numpy.transpose(b_quad))
            numpy.save('./fnodes_quadratic.npy', f_quad)
            print("Number of quadratic basis elements is ", ndim_quad, "and the linear ROQ data save in B_quadratic.npy")
            break
    return

def testrep_quad(b_quad, emp_nodes_quad, test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant):
    hp_test_quad = (numpy.absolute(generate_a_waveform_from_mcq(test_mc_quad, test_q_quad, test_s1_quad, test_s2_quad, test_ecc_quad, test_lambda1_quad, test_lambda2_quad, test_iota_quad, test_phiref_quad, distance, deltaF, f_min, f_max, waveFlags, approximant)))**2
    hp_test_quad_emp = hp_test_quad[emp_nodes_quad]
    hp_rep_quad = numpy.dot(b_quad,hp_test_quad_emp)
    diff_quad = hp_rep_quad - hp_test_quad
    rep_error_quad = diff_quad/numpy.vdot(hp_test_quad,hp_test_quad)**0.5
    plt.plot(numpy.real(rep_error_quad))
    plt.xlabel('Waveform Node Number')
    plt.ylabel('Fractional Representation Error for Quadratic')
    plt.title('Rep Error with numpy.linalg.inv()')
    plt.show()
    return

def surros_of_test_samples(nsamples, nparams, params_low, params_high, tolerance, b_linear, emp_nodes, distance, deltaF, f_min, f_max, waveFlags, approximant):
    nts=nsamples
    ndim = len(emp_nodes)
    test_points = generate_params_points(nts, nparams, params_low, params_high)
    surros = numpy.zeros(nts)
    for i in numpy.arange(0,nts):
        test_mc =  test_points[i,0]
        test_q = test_points[i,1]
        test_s1 = spherical_to_cartesian(test_points[i,2:5])
        test_s2 = spherical_to_cartesian(test_points[i,5:8])
        test_iota = test_points[i,8]
        test_phiref = test_points[i,9]
        test_ecc = 0
        test_lambda1 = 0
        test_lambda2 = 0
        if nparams == 11: test_ecc = test_points[i,10]
        if nparams == 12: 
            test_lambda1 = test_points[i,10]
            test_lambda2 = test_points[i,11]
        hp_test = generate_a_waveform_from_mcq(test_mc, test_q, test_s1, test_s2, test_ecc, test_lambda1, test_lambda2, test_iota, test_phiref, distance, deltaF, f_min, f_max, waveFlags, approximant)
        hp_test_emp = hp_test[emp_nodes]
        hp_rep = numpy.dot(b_linear,hp_test_emp) 
        surros[i] = (1-overlap_of_two_waveforms(hp_test, hp_rep))*deltaF
    if (surros[i] > tolerance):
        print("iter", i, surros[i], test_points[i])
    if i%100==0:
        print("iter", i, surros[i])
    return surros