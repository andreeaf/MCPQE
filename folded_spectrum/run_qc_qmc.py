import hamil
import utils
import encode_ucc_wavefunction as euw
import measurements

from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner

import numpy as np
import cProfile
import pstats
import time
from sympy import Symbol
from pytket.pauli import Pauli
from pytket.circuit import Circuit

import concurrent
import copy

from pytket.circuit import Circuit
def count_set_bits(n):
    m = n
    count = 0
    while m:
        if m & 1:
            count += 1
        m = m >> 1
    return count
def excit_to_det(occ_det, occ_ref, nqubits):
    det = sum(2**(nqubits-1-i) for i in occ_det)
    ref = sum(2**(nqubits-1-i) for i in occ_ref)
    #det = sum(2**i for i in occ_det)
    #ref = sum(2**i for i in occ_ref)
    excit = det^ref
    excit_level = count_set_bits(excit)/2
    nannih = excit_level
    ncreate = excit_level

    sign = 1

    for ibit in range(63, -1, -1):
        if (ref & 2**ibit):
            if (excit & 2**ibit):
                nannih -= 1
            elif (nannih+ncreate)%2 == 1:
                sign = -sign
        elif excit & 2**ibit:
            ncreate -= 1
    return sign
#with cProfile.Profile() as pr:
print(np.linspace(1.0, 2.5, 16))
#omega = {(0,1,2):0, (0,2,5): 0.9, (0,3,4):1.1, (1,2,4): 0.6, (2,4,5):1.75}
#omega = {(0,2,3):0.25, (0,1,4):0.6, (2,3,4):1.1, (0,4,5): 1.6}
dets = [(0,3), (1,2), (3,4), (2,5)]
#dets = [(0,1), (0,5), (1,4), (2,3), (4,5)] 
#omega = {(0,3,4):1.1}
#stoch_hamil = False
for nshots in [100]:#, 5000, 1000, 100, 50, 10]:
    gaussian = False
    #omega = {(0, 1): 0, (1, 4): 1.0, (0, 5): 0.75, (2,3):0.5, (4,5):1.5}
    #omega = {(0,1,2):0, (0,2,5): 0.9, (0,3,4):1.1, (1,2,4): 0.6, (2,4,5):1.75}
    #omega = {(0,2,3):0.25, (0,1,4):0.6, (2,3,4):1.1, (0,4,5): 1.6}
    omega = {(0,3):0.15, (1,2):0.35, (3,4):1.0, (2,5):1.4}
    #omega = {(0,1,2,3):0, (0,2,3,5):0.45, (1,2,3,4):0.55, (0,1,4,5):0.8, (0,1,3,6):0.9, (1,3,4,6):0.95, (0,1,2,7):1.01, (2,3,4,5):1.2} 

    #dets = list(omega.keys()) + [(1,2,5,6), (0,1,6,7), (1,2,4,7), (0,3,5,6), (0,3,4,7), (2,3,6,7), (0,2,5,7)]
    nqubits = 6
    pops = {d:0 for d in dets}
    omega_pops = {ref:pops.copy() for ref in omega}
    for ref in omega:
        omega_pops[ref][ref] = 1 
    for r in np.linspace(1.0, 2.5, 31):
        geometry = [['H', [0,0,0]], ['H', [0,0,r]], ['H', [0,0,2*r]]]#, ['H', [0,0,3.0]]]
        basis = 'sto-3g'
        multiplicity = 1 
        charge = 1
        hf_e, fci_e, true_qubit_h = hamil.get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
        true_ps_H = hamil.get_hamil_pauli_strings(true_qubit_h)
        true_groups = utils.group_pauli_strings(true_ps_H)
        true_mats = hamil.get_hamil_group_matrices(true_groups, nqubits, 'test')
        true_hmat_comp = np.zeros((2**7,2**7), dtype=np.complex128)
        for string in true_ps_H:
            circ = Circuit(7)
            for i in range(len(string[0])):
                if string[0][i] == Pauli.Z:
                    circ.Z(string[2][i])
                if string[0][i] == Pauli.X:
                    circ.X(string[2][i])
                if string[0][i] == Pauli.Y:
                    circ.Y(string[2][i])
            mat = string[1] * circ.get_unitary()
            true_hmat_comp += mat
        true_hmat = np.zeros((2**7,2**7))
        for i in range(2**7):
            for j in range(2**7):
                true_hmat[i,j] = true_hmat_comp[i,j].real
        for ref in [(3,4)]:
            stoch_hamil = False #True 
            hf_e, fci_e, qubit_h = hamil.get_qubit_molecular_hamiltonian_squared(geometry, basis, multiplicity, charge, omega=omega[ref])
            ps_H = hamil.get_hamil_pauli_strings(qubit_h)
            hmat_comp = np.zeros((2**7,2**7), dtype=np.complex128)
            for string in ps_H:
                circ = Circuit(7)
                for i in range(len(string[0])):
                    if string[0][i] == Pauli.Z:
                        circ.Z(string[2][i])
                    if string[0][i] == Pauli.X:
                        circ.X(string[2][i])
                    if string[0][i] == Pauli.Y:
                        circ.Y(string[2][i])
                mat = string[1] * circ.get_unitary()
                hmat_comp += mat
            circ = Circuit(7)
            circ.Z(0)
            zmat_comp = circ.get_unitary()
            zmat = np.zeros((2**7,2**7))
            hmat = np.zeros((2**7,2**7))
            for i in range(2**7):
                for j in range(2**7):
                    hmat[i,j] = hmat_comp[i,j].real
                    zmat[i,j] = zmat_comp[i,j].real
            
            dets.sort(key = lambda x: utils.dets_list_order(x, ref))
            mapping = {det: utils.get_bits(det, nqubits +1) for det in range(2**(nqubits+1))}
            signs = {d:excit_to_det(d,ref, nqubits) for d in dets}
            operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
            qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
            params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
            ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
            ucc_circ2 = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
            pops = omega_pops[ref]
            #norm = sum([abs(x) for x in pops.values()])
            #for det in dets:
            #    pops[det] /= norm
            
            groups = utils.group_pauli_strings(ps_H)
            mats = hamil.get_hamil_group_matrices(groups, nqubits, 'test')
            if stoch_hamil:
                group_weights = {key:sum(abs(groups[key][i][1]) for i in range(len(groups[key]))) for key in groups}
                keys = list(groups.keys())
                group_weights_cumul = {}
                for i in range(1, len(keys)):
                    group_weights_cumul[keys[i]] = sum(group_weights[keys[i]] for i in range(1,i+1))
                tot_weight = sum(group_weights[keys[i]] for i in range(1,i+1))
            sym = [0, 0, 1, 1, 0, 0, 1, 1]
            xi = 0.1
            
            dt = 0.2
            for pop_thresh in [1.5]:
                t1 = time.time()
                S = omega[ref]**2 + 0.1
                step_0 = 0
                pop_old = 1
                old_e = 0
                e_proj = 0
                old_shift = 0
                vary_shift = False
                f = open('34/H3+.sym1.'+'{0:.2f}'.format(r)+'A.sto-3g.'+str(nshots)+'shots.folded_spectrum.omega=' + str(omega[ref]) +'.out', 'w')
                print(r, ref)
                print('Starting QMC \n')
                f.write('FCI energy: ' + str(fci_e)+'\n')
                f.write('HF energy: ' + str(hf_e)+'\n')
                f.write('delta t = ' + str(dt)+'\n')
                f.write('ref = ' + str(ref)+'\n')
                spawned_dets = []
                new = False
                sym_ref = 0
                stoch_wfn = False #True 
                nattempts = 10
                told=t1
                for step in range(1000):
                    pop_change = {}
                    if stoch_wfn:
                        pops_loc = [(x, pops[x]) for x in dets[1:]]
                        pops_loc.sort(key=lambda x: -abs(x[1]))
                        if len(dets)>1:
                            ms = sum(abs(x) for x in [pops[x] for x in dets[1:]])
                        else:
                            ms = 0
                        rand = np.random.rand()*ms
                        param_map = {}
                        s = 0
                        for term in pops_loc:
                            if s < rand:
                                param_map[params[term[0]]] = term[1]/pops[ref]
                                s +=  abs(term[1])
                            else:
                                param_map[params[term[0]]] = 0
                    else:
                        param_map = {params[d]: pops[d]/pops[ref] for d in params}
                    current_ucc_circ = ucc_circ.copy()
                    current_ucc_circ.symbol_substitution(param_map)
                    resids = {}
                    if stoch_hamil:
                        groups_loc = {keys[0]:groups[keys[0]]}
                        for attempt in range(nattempts):
                           rand = np.random.rand()*tot_weight
                           for grp in group_weights_cumul:
                               if rand<group_weights_cumul[grp]:
                                   if grp in groups_loc:
                                       for g in range(len(groups_loc[grp])):
                                           groups_loc[grp][g][1] += groups[grp][g][1]/nattempts/group_weights[grp]*tot_weight
                                   else:
                                       groups_loc[grp] = copy.deepcopy(groups[grp])
                                       for g in groups_loc[grp]:
                                           g[1]*=tot_weight/(group_weights[grp])/nattempts
                                   break                

                    for det in dets:
                        mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                        if stoch_hamil:
                            hamil_term = measurements.measure_hamil_overlap_groups_matrices(groups_loc, mats, mcirc, mapping, nshots, gaussian)
                        else:
                            hamil_term = measurements.measure_hamil_overlap_groups_matrices(groups, mats, mcirc, mapping, nshots, gaussian)
                        overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, nshots, gaussian)
                        if step == 0 and ref == det:
                             S = hamil_term/overlap
                        hamil_term -= S * overlap
                        resids[det] = hamil_term * signs[det]
                        if det in pop_change:
                            pop_change[det] -= dt* pops[ref]*hamil_term*signs[det]
                        else:
                            pop_change[det] = -dt* pops[ref]*hamil_term*signs[det]
                    for det in dets:
                        pops[det] += pop_change[det]
                    if vary_shift and (step - step_0) % 5 == 0:
                        old_shift = S
                        S = S - xi/(5*dt)*np.log(sum([abs(pops[x]) for x in pops])/pop_old)
                        pop_old = sum([abs(pops[x]) for x in pops])
                
                    if sum([abs(pops[x]) for x in pops]) >= pop_thresh and not vary_shift:
                        mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                        hamil_term = measurements.measure_hamil_overlap_groups_matrices(groups, mats, mcirc, mapping, nshots, gaussian)
                        overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, nshots, gaussian)
                        S = hamil_term/overlap
                        vary_shift = True
                        print('Shift varying')
                        step_0 = step
                        pop_old = sum([abs(pops[x]) for x in pops])
                    mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                    old_e = e_proj
                    hamil_term = measurements.measure_hamil_overlap_groups_matrices(groups, mats, mcirc, mapping, nshots, gaussian)
                    overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, nshots, gaussian)
                    e_proj = hamil_term/overlap
                    hamil_term = measurements.measure_hamil_overlap_groups_matrices(true_groups, true_mats, mcirc, mapping, nshots, gaussian)
                    true_e_proj = hamil_term/overlap
                
                    if step%1 == 0: 
                        f.write(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str (true_e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                        if step%5==0:
                            print(step, true_e_proj, true_e_proj - hf_e, abs(S-old_shift), abs(e_proj - old_e))
                            if vary_shift and abs(S-old_shift) < 1e-5 and abs(e_proj - old_e) < 1e-5:
                                hamil_term = measurements.measure_hamil_overlap_groups_matrices(true_groups, true_mats, mcirc, mapping, nshots)
                                overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, nshots)
                                true_e_proj = hamil_term/overlap
                                print(step)
                                break
                    omega_pops[ref] = pops.copy()
                f.write('Corresponding energy: ' +str(true_e_proj)+'\n')
                f.write('Elapsed time: '+str(time.time()-t1)+ ' seconds')
                f.close()
                a = np.genfromtxt('34/H3+.sym1.'+'{0:.2f}'.format(r)+'A.sto-3g.'+str(nshots)+'shots.folded_spectrum.omega=' + str(omega[ref]) +'.out', unpack = True, usecols = (3,), skip_header=4, skip_footer=2) 
                s = 0
                n = 0
                print(step_0, len(a))
                for i in range(step_0, len(a)):
                    if a[i] > -10000 and a[i] < 10000:
                        s += a[i]
                        n +=1
                omega[ref] = s/n - hf_e#np.average(a)#true_e_proj#*0.99
                print('Moving on to next calculation')
            print('omega', omega)
