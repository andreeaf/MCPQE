import numpy as np
import math
import json
import argparse

def count_set_bits(n):
    m = n
    count = 0
    while m:
        if m & 1:
            count += 1
        m = m >> 1
    return count

def check_excitation_possible(excit, det, ref):
    annihilation = excit & ref
    creation = excit - (excit & ref)
    det_annih = (det ^ ref) & ref
    det_creation = (det ^ ref) - ((det ^ ref) & ref)    
    if (creation & det_creation != 0 or annihilation & det_annih != 0):
        possible = False
    else:
        possible = True
    return possible
def check_deexcitation_possible(excit, det, ref):
    creation = excit & ref
    annihilation = excit - (excit & ref)
    det_annih = (det ^ ref) & ref
    det_creation = (det ^ ref) - ((det ^ ref) & ref)    
    possible = True
    for n in range(64):
        if (annihilation & (1<<n) and not det_creation & (1<<n)):
            possible = False
            break
        if (creation & (1<<n) and det & (1<<n)):
            possible = False
            break
    return possible   
def excite(det, excit, ref):
    sign = 1
    new_det = det
    det_annih = (det ^ ref) & ref
    det_creation = (det ^ ref) - det_annih

    for i in range(64):
        mask = (~0)<<i+1
        if excit & 1<<i:
            if ref & 1<<i:
            # If exciting from orbital i, permute with all annihilation of
            # higher index and add creation
               new_det = new_det & ~(1<<i)
               permute = mask & det_annih
               permute = permute ^ det_creation
            else:
            # If exciting into orbital i, permute with every creation of lower index
               new_det = new_det | (1<<i)
               permute =  ~mask & det_creation 
               permute = permute & ~(1<<i)
            if (count_set_bits(permute)%2 == 1):
                sign = -sign
    return new_det, sign

def deexcite(det, excit, ref):
    sign = 1
    new_det = det
    det_annih = (det ^ ref) & ref
    det_creation = (det ^ ref) - det_annih
    excit_annih = excit & ~ref
    for i in range(63, -1, -1):
        mask = ~0<<i+1
        if excit & 1<<i:
            if not (ref & 1<<i):
               #If exciting from i, permute with all creation of lower index
               new_det = new_det & ~(1<<i)
               permute =  ~mask & det_creation
               excit_swap = ~mask & excit_annih
               permute = permute ^ excit_swap
               permute = permute & ~(1<<i)
            else:
               #If exciting into i, permute with all creation operators and annihilation of higher index
               new_det = new_det | (1<< i)
               permute = mask & det_annih
               permute = permute ^ det_creation
            if (count_set_bits(permute)%2 == 1):
               sign = -sign
            det_annih = (new_det ^ ref) & ref
            det_creation = (new_det ^ ref) & ~ref
    return new_det, sign

def excit_to_det(det, ref):
    excit = det^ref
    excit_level = count_set_bits(excit)/2
    nannih = excit_level
    ncreate = excit_level

    sign = 1

    for ibit in range(64):
        if (ref & 2**ibit):
            if (excit & 2**ibit):
                nannih -= 1
            elif (nannih+ncreate)%2 == 1:
                sign = -sign
        elif excit & 2**ibit:
            ncreate -= 1
    return sign

def get_proj_energy(final_contrib, keys, hmat, ref):
    normalisation = 0
    proj_energy = 0
    normalisation = final_contrib[ref]
    for j in keys:
        if j in final_contrib:
            proj_energy += final_contrib[j]*hmat[j][ref]
    proj_energy /= normalisation
    return proj_energy

def get_var_energy_and_resid_proj(total_contrib, keys, active_keys, hmat, proj_energy):
    residuals_proj_it = {}
    var_energy = 0
    normalisation = 0
    for key1 in active_keys:
        normalisation +=total_contrib[key1]**2
        residual_proj = 0
        for key2 in active_keys:  
            if key1 in hmat and key2 in hmat: 
                var_energy += total_contrib[key1]*total_contrib[key2]*hmat[key1][key2]
            if key1 in keys:
                if key1 == key2:
                        residual_proj += total_contrib[key2]*(hmat[key1][key2]-proj_energy)
                else:
                        residual_proj += total_contrib[key2]*(hmat[key1][key2])
        if key1 in keys:
            residuals_proj_it[key1] = residual_proj
    var_energy /= normalisation
    return var_energy, residuals_proj_it

def get_ci_wfn(coeffs, excitors, keys, ordered_keys, ref):
    new_keys=keys.copy()
    ts = coeffs
    contributions = {}

    for i in range(len(keys)):
        contributions[i] = {} 
    contributions[0][ref] = ts[ref]
    
    for i in range(1,len(keys)):
        excit = ordered_keys[i]
        for det in contributions[i-1]:
            possible1 = check_excitation_possible(excitors[excit], det, ref)
            if possible1:
                new_det, sign = excite(det, excitors[excit], ref)
                if new_det not in new_keys:
                    new_keys.append(new_det)
                if new_det in contributions[i]:
                    contributions[i][new_det] += contributions[i-1][det]*np.sin(ts[excit])*sign
                else:
                    contributions[i][new_det] = contributions[i-1][det]*np.sin(ts[excit])*sign
            possible2 = check_deexcitation_possible(excitors[excit], det, ref)
            if possible2:
                new_det, sign = deexcite(det, excitors[excit], ref)
                if new_det not in new_keys:
                    new_keys.append(new_det)
                if new_det in contributions[i]:  
                    contributions[i][new_det] -= contributions[i-1][det]*np.sin(ts[excit])*sign
                else:
                    contributions[i][new_det] = -contributions[i-1][det]*np.sin(ts[excit])*sign
            if possible1 or possible2:
                if det in contributions[i]:
                    contributions[i][det] += contributions[i-1][det]*np.cos(ts[excit])
                else:
                    contributions[i][det] = contributions[i-1][det]*np.cos(ts[excit])
            else:
                if det in contributions[i]:
                    contributions[i][det] += contributions[i-1][det]
                else:
                    contributions[i][det] = contributions[i-1][det]
    final_contrib = {}   
    for j in new_keys:
        if j in contributions[len(keys)-1]:
            final_contrib[j] = contributions[len(keys)-1][j]*excit_to_det(j, ref)
    return new_keys, final_contrib

def read_in_ci_hamiltonian(filename, ref):

    keys = []
    hmatel = np.genfromtxt(filename)
    hmat = {}
    for hel in hmatel:
        if int(hel[0]) not in keys:
            keys.append(int(hel[0]))
        if int(hel[0]) not in hmat:
            hmat[int(hel[0])] = {}
        if int(hel[1]) not in hmat:
            hmat[int(hel[1])] = {}
        hmat[int(hel[0])][int(hel[1])] = hel[2]
        hmat[int(hel[1])][int(hel[0])] = hel[2]
    #for hel in hmatel:
    #    if hel[0]==hel[1] and int(hel[0])!=ref:
    #        hmat[int(hel[1])][int(hel[0])] -= hmat[ref][ref]
    #hmat[ref][ref]=0
    return hmat

    
def run_tucc(filename, ref, active_keys, ordered_keys, tau = 0.01, convergence = 1e-6, debug = False):
    hmat = read_in_ci_hamiltonian(filename, ref)
    coeffs = {}
    keys = []

    coeffs[ref] = 1.0
    for key in active_keys:
        if key != ref:
            coeffs[key] = 0.0
    excitors={}
    for i in range(len(active_keys)):
        excitors[active_keys[i]] = active_keys[i] ^ ref

    k = 0
    keys, wfn = get_ci_wfn(coeffs, excitors, active_keys, ordered_keys, ref)
    proj_energy = get_proj_energy(wfn, keys, hmat, ref)
    var_energy, residuals_proj = get_var_energy_and_resid_proj(wfn, active_keys, keys, hmat, proj_energy)

    abs_resid = {}
    for key in active_keys:
        abs_resid[key] = abs(residuals_proj[key])

    f = open(filename+'tucc_solve.dat', 'w')
    f.write('Reference = ' + str(ref) + '\n')
    while max(abs_resid.values()) > convergence:
       old_abs = max(abs_resid.values())
       if(debug == True and k%10==0):
          print(old_abs)
       k +=1
       for key in active_keys[1:]:
           coeffs[key] -= tau*residuals_proj[key]*excit_to_det(key, ref)
           print(key, ref, excit_to_det(key, ref))
       keys, wfn = get_ci_wfn(coeffs, excitors, active_keys, ordered_keys, ref)
       proj_energy = get_proj_energy(wfn, keys, hmat, ref)
       var_energy, residuals_proj = get_var_energy_and_resid_proj(wfn, active_keys, keys, hmat, proj_energy)
       for key in active_keys:
           abs_resid[key] = abs(residuals_proj[key])
       print(proj_energy, max(abs_resid.values()))
       f.write(str(k) + '    '+str(proj_energy)+'    '+str(max(abs_resid.values()))+ '    ' + str(coeffs) +'\n')
    f.write('Final wfn:   ' + str(wfn) )
    print(wfn)
    f.close()
    return coeffs, proj_energy, var_energy

parser = argparse.ArgumentParser(description='Run tUCC for a given system')

parser.add_argument('--hamil', type = str, help = 'File containing the CI Hamiltonian, as output by HANDE')
parser.add_argument('--ref', type = int, help = 'Bit string encoding the reference determinant')
parser.add_argument('--basis', type = int, help = 'Number of spatial orbitals in basis set')

args = parser.parse_args()
ref = args.ref
basis = args.basis
hamil = args.hamil
nelec = count_set_bits(ref)

print('Running calculation for a system with ' + str(nelec) + ' electrons in ' + str(basis) + ' spatial orbitals.')
coeffs, proj_energy,var_energy = run_tucc(hamil, ref, [3,12, 18,33,48], [3,33,12,48,18])
print(proj_energy)
