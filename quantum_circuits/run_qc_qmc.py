import hamil
import utils
import encode_ucc_wavefunction as euw

from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner

import numpy as np
import cProfile
import pstats
import time
from sympy import Symbol

import concurrent
import copy
def count_set_bits(n):
    m = n
    count = 0
    while m:
        if m & 1:
            count += 1
        m = m >> 1
    return count
def excit_to_det(occ_det, occ_ref):
    det = sum(2**i for i in occ_det)
    ref = sum(2**i for i in occ_ref)
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
geometry = [['H', [0,0,0]], ['H', [0,0,1.0]], ['H', [0,0,2.0]]]#, ['H', [0,0,3.0]]]
basis = 'sto-3g'
multiplicity = 1
charge = 1

hf_e, fci_e, qubit_h = hamil.get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
ps_H = hamil.get_hamil_pauli_strings(qubit_h)
groups = utils.group_pauli_strings(ps_H)
group_weights = {key:sum(abs(groups[key][i][1]) for i in range(len(groups[key]))) for key in groups}
keys = list(groups.keys())
group_weights_cumul = {}
for i in range(1, len(keys)):
    group_weights_cumul[keys[i]] = sum(group_weights[keys[i]] for i in range(1,i+1))
    tot_pop = sum(group_weights[keys[i]] for i in range(1,i+1))

sym = [0, 0, 1, 1, 0, 0, 1, 1]
#dets = [(0,1,2,3), (1, 2, 3, 4), (2,3, 4,5), (2,3,6,7), (1,3,4,6), (1,2,5,7), (0,2,3,5), (0,3,4,7), (0,3,5,6), (0,2,5,7), (0,1,3,6), (0,1,4,5), (0,1,6,7), (0,1,2,7)]
#dets = [(0,1,2), (1,2,4), (2,4,5), (0,2,5), (0,3,4)]
#dets = [(0,1), (0,5), (2,3), (4,5), (1,4)]
dets = [(0,1), (2,3), (4,5), (0,5), (1,4)]
#dets = [(0,1,2,3)]
ref = (0,1)
dets.sort(key = lambda x: utils.dets_list_order(x, ref))
signs = {d:excit_to_det(d,ref) for d in dets}
xi = 1
nqubits = 6 
operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
all_circs = {}
overlap_circs = {}
for det in dets:
    mcirc1 = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, ucc_circ)
    circs = utils.generate_hamil_overlap_measure_circuits(mcirc1, groups, nqubits)
    all_circs[det] = circs
    overlap_circs[det] = utils.generate_overlap_measurement(mcirc1)
ucc_circ2 = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
apply_circs = utils.generate_hamil_apply_circ(ucc_circ2, groups, nqubits)

for nshots in [1000]:#, 5000, 1000, 100, 50, 10]:
    dt = 0.05
    for pop_thresh in [2]:
        t1 = time.time()
        pops = {d:0 for d in dets[1:]}
        pops[ref] = 1 
        S = 0
        step_0 = 0
        pop_old = 1
        vary_shift = False
        f = open('H3+.sto3g.1A.sv-test_stoch_hamil_5_importance.out', 'w')
        print('Starting QMC \n')
        f.write('FCI energy: ' + str(fci_e)+'\n')
        f.write('HF energy: ' + str(hf_e)+'\n')
        f.write('delta t = ' + str(dt)+'\n')
        #with cProfile.Profile() as profile:
        spawned_dets = []
        new = False
        sym_ref = 0
        told=t1
        for step in range(1000):
            pop_change = {}
            param_map = {params[d]: pops[d]/pops[ref] for d in params}
            #print(rand, param_map)
            current_ucc_circ = ucc_circ.copy()
            current_ucc_circ.symbol_substitution(param_map)
            #if step%len(keys)==0:
            #    if step != 0:
            #        for det in dets:
            #            pops[det] += pop_change[det]
            #    pop_change = {}
            #ms = round(max(abs(x) for x in [pops[x] for x in dets[1:]]),1)
            #rand = np.random.rand()*ms
            #param_map = {params[d]: pops[d]/pops[ref] if abs(pops[d])>rand else 0 for d in params}
            
            groups_loc = {keys[0]:groups[keys[0]]}
            #keys_loc = []
            for attempt in range(5):
                #key = int(np.floor(np.random.rand()*len(keys)))
                #keys_loc.append(key)
                #groups_loc[keys[key]] = groups[keys[key]]
                rand = np.random.rand()*tot_pop
                for grp in group_weights_cumul:
                    if rand<group_weights_cumul[grp]:
                        if grp in groups_loc:
                            for g in range(len(groups_loc[grp])):
                                groups_loc[grp][g][1] += groups[grp][g][1]/group_weights[grp]*tot_pop/5
                        else:
                            groups_loc[grp] = copy.deepcopy(groups[grp])
                            for g in groups_loc[grp]:
                                g[1]/=5*group_weights[grp]/tot_pop
                        break                
            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                ##mcirc.to_latex_file(str(step)+str(det)+'.tex')
                hamil_term = pops[ref]*(euw.measure_hamil_overlap(mcirc, groups_loc, nqubits)) 
                #if 0 in keys_loc: 
                hamil_term -= pops[ref]*(hf_e + S) * utils.measure_overlap(mcirc)
                    #pops[det] -= dt* hamil_term2*signs[det]
                #circs_num = {}
                #for key in all_circs[det]:
                #    circ = all_circs[det][key].copy()
                #    circ.symbol_substitution(param_map)
                #    circs_num[key] = circ
                ##hamil_term = pops[0]*(euw.measure_hamil_overlap(mcirc, groups, nqubits, True, nshots) -
                ##                      S * utils.measure_overlap(mcirc, True, nshots))
                #overlap_circ = overlap_circs[det].copy()
                #overlap_circ.symbol_substitution(param_map)
                #ho, counts = utils.measure_hamil_overlap_shots(groups, circs_num, nshots)
                ##for ct in counts:
                ##    if ct not in spawned_dets:
                ##        spawned_dets.append(ct)
                #hamil_term = pops[ref]*(ho - (S + hf_e) * utils.measure_overlap_shots(overlap_circ, nshots))
                if det in pop_change:
                    pop_change[det] -= dt* hamil_term*signs[det]
                else:
                    pop_change[det] = -dt* hamil_term*signs[det]
           # for group, circ in apply_circs.items():
           #     circ_num = circ.copy()
           #     circ_num.symbol_substitution(param_map)
           #     counts = utils.get_spawns(circ_num)
            #    for count in counts:
            #        if count not in spawned_dets:
            #            spawned_dets.append(count)
            #dets, new = utils.generate_new_dets_list(dets, spawned_dets, nqubits, sym, sym_ref)
            #if new:
            #    dets.sort(key = lambda x: utils.dets_list_order(x, ref))
            #    print(dets)
            #    for det in dets:
            #        if det not in pops:
            #            pops[det] = 0
            #    new = False
            #    print(105)
            #    operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
            #    qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
            #    params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
            #    print(109)
            #    ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
            #    all_circs = {}
            #    overlap_circs = {}
            #    print(113)
            #    for det in dets:
            #        print(det)
            #        mcirc1 = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, ucc_circ)
            #        circs = utils.generate_hamil_overlap_measure_circuits(mcirc1, groups, nqubits)
            #        all_circs[det] = circs
            #        overlap_circs[det] = utils.generate_overlap_measurement(mcirc1)
            #    print(120)
            #    ucc_circ2 = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
            #    apply_circs = utils.generate_hamil_apply_circ(ucc_circ2, groups, nqubits)
            #    param_map = {params[d]: pops[d]/pops[ref] for d in params}
            #    print(124)
            for det in dets:
                pops[det] += pop_change[det]
            if vary_shift and (step - step_0) % 5 == 0:
                S -= xi/(5*dt)*np.log(sum([abs(pops[x]) for x in pops])/pop_old)
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if sum([abs(pops[x]) for x in pops]) >= pop_thresh and not vary_shift:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                S = euw.measure_hamil_overlap(mcirc, groups, nqubits) - hf_e * utils.measure_overlap(mcirc)# True, nshots)
                #circs_num = {}
                #for key in all_circs[ref]:
                #    circ = all_circs[ref][key].copy()
                #    circ.symbol_substitution(param_map)
                #    circs_num[key] = circ
                #S = utils.measure_hamil_overlap_shots(groups, circs_num, nshots)[0] - hf_e * utils.measure_overlap_shots(overlap_circ, nshots)
                vary_shift = True
                step_0 = step
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if step%5 == 0: 
                f.write(str(step)+'    '+str (S)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                print(str(step)+'    '+str (S)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                #t2 = time.time()
                #print(t2-told) 
                #told = t2
        t2 = time.time()
        f.write('Elapsed time: '+str(t2-t1)+ ' seconds')
        f.close()
        #ps = pstats.Stats(profile)
        #ps.sort_stats('cumtime')
        #ps.print_stats(10)
        print('Moving on to next calculation')
