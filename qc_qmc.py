from openfermion import jordan_wigner
import utils
import encode_ucc_wavefunction as euw
from sympy import Symbol
import time
import hamil
import numpy as np
import copy
from openfermion import FermionOperator, hermitian_conjugated, normal_ordered
from scipy.sparse.linalg import eigsh
import stoch_gates 

def get_stoch_wfn(pops, params, noise_symbols = None, p = None, gamma = None):

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
            if noise_symbols is not None:
                param_map.update(stoch_gates.get_noise(noise_symbols[term[0]], p, gamma))
        else:
            param_map[params[term[0]]] = 0
            if noise_symbols is not None:
                for symbol in noise_symbols[term[0]]:
                    param_map[symbol] = 0
                
    return param_map

def get_stoch_hamil(keys, groups, nattempts, tot_weight, group_weights_cumul):
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
    return groups_loc

def spawn(apply_mats, ucc_circ2, param_map, dets, nqubits, ref):

    for group, mat in apply_mats.items():
        circ_num = ucc_circ2.copy()
        circ_num.symbol_substitution(param_map)
        counts = utils.get_counts(np.dot(mat, circ_num.get_statevector()), 10, mapping)
        for count in counts:
            if count not in spawned_dets:
                spawned_dets.append(count)
    dets, new = utils.generate_new_dets_list(dets, spawned_dets, nqubits)
    if new:
        dets.sort(key = lambda x: utils.dets_list_order(x, ref))
        for det in dets:
            if det not in pops:
                pops[det] = 0
        new = False
        operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
        qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
        params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
        ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
        param_map = {params[d]: pops[d]/pops[ref] for d in params}
        signs = {d:utils.excit_to_det(d,ref) for d in dets}
    return operators, qubit_ops, params, ucc_circ, param_map, signs

def qc_qmc_ucc(ferm_hamil, ref, hf_e, nqubits, dt, pop_thresh, filename, nsteps = 100, dets = None, stoch_hamil = False, sym = None, shots = False, nattempts = 1, stoch_wfn = False,hmat = None,zmat = None, mats = None, apply_mats=None, depolarising_noise=False, noise_prob=0, gaussian_noise=False, damping_noise=False, damp_prob=0):
    
    #Map integers to occupation strings of the whole system + ancilla Hilbert space
    mapping = {det: utils.get_bits(det, nqubits +1) for det in range(2**(nqubits+1))}
    t1 = time.time()

    #Decide if spawning is necessary
    spawning = False
    if dets is None:
        spawning = True
        dets = [ref]
    if damp_prob:
        damp_prob = 2 * np.arcsin(np.sqrt(damp_prob))/np.pi

    #Transform Hamiltonian into qubit basis and group 
    qubit_h = jordan_wigner(ferm_hamil) 
    ps_H = hamil.get_hamil_pauli_strings(qubit_h)
    groups = utils.group_pauli_strings(ps_H)
    if hmat is None or zmat is None or mats is None:
        hmat, zmat = hamil.get_hamil_and_overlap_matrix(ps_H, nqubits, 'test')
        mats = hamil.get_hamil_group_matrices(groups, nqubits, 'test')
    evals, evecs = eigsh(hmat)

    #Generate probabilities for Hamiltonian term sampling
    if stoch_hamil:
        group_weights = {key:sum(abs(groups[key][i][1]) for i in range(len(groups[key]))) for key in groups}
        keys = list(groups.keys())
        group_weights_cumul = {}
        for i in range(1, len(keys)):
            group_weights_cumul[keys[i]] = sum(group_weights[keys[i]] for i in range(1,i+1))
        tot_weight = sum(group_weights[keys[i]] for i in range(1,i+1))

    #Sort determinants according to "standard" order
    dets.sort(key = lambda x: utils.dets_list_order(x, ref))
    xi = 1 #TODO: Pass in as parameter

    #Generate excitation operators corresponding to determinants
    operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
    qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
    params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}

    #Generate corresponding UCC ansatz and all measurement circuits
    if depolarising_noise or damping_noise:
        noise_symbols = {}
        ucc_circ = stoch_gates.generate_noisy_controlled_ucc_circ(nqubits, qubit_ops, params, noise_symbols, depolarising_noise = depolarising_noise, amplitude_damping = damping_noise)
        print(noise_symbols)
    else:
        ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
    if spawning:
        #Generate spawning circuits.
        ucc_circ2 = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
        if apply_mats is None:
            apply_mats = hamil.get_hamil_matrix(ps_H, nqubits)
        new_spawned_dets = False
        spawned_dets = []

    #Initialise QMC 
    signs = {d:utils.excit_to_det(d,ref) for d in dets}
    pops = {d:0 for d in dets[1:]}
    pops[ref] = 1 
    S = 0
    old_S = 1
    step_0 = 0
    pop_old = 1
    vary_shift = False

    #Open output file
    f = open(filename, 'w')
    print('Starting QMC \n')
    f.write('step      S         E_proj        N0         N_tot       Ni/N0\n')
    e_proj = 0


    #Do QMC
    if (not shots) and gaussian_noise:
        print('Invalid parameters')
        exit(
)
    if shots and not gaussian_noise:
        for step in range(nsteps+1):
        
            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, noise_symbols, prob_noise, damp_noise) 
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params) 
            elif depolarising_noise or damping_noise:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 
                for d in params:
                    param_map.update(stoch_gates.get_noise(noise_symbols[d], noise_prob, damp_prob))
            else:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 

            current_ucc_circ = ucc_circ.copy()
            current_ucc_circ.symbol_substitution(param_map)
            pop_change = {}

            if stoch_hamil:
                groups_loc = get_stoch_hamil(keys, groups, nattempts, tot_weight, group_weights_cumul)

            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                if stoch_hamil:
                    ho = utils.measure_hamil_overlap_groups_matrices(groups_loc, mats, mcirc, mapping, shots)
                else:
                    ho = utils.measure_hamil_overlap_groups_matrices(groups, mats, mcirc, mapping, shots)
                overlap = utils.measure_overlap_shots_mat(mcirc, mapping, shots)
                hamil_term = pops[ref]*(ho - (S + hf_e) * overlap)
                pop_change[det] = -dt* hamil_term*signs[det]
                if det == ref:
                    e_proj = ho/overlap - hf_e

            for det in dets:
                pops[det] += pop_change[det]
            if spawning:
                operators, qubit_ops, params, ucc_circ, param_map, sign = spawn(apply_mats, ucc_circ2, param_map, dets, nqubits, ref)
    
            if vary_shift and (step - step_0) % 5 == 0:
                S -= xi/(5*dt)*np.log(sum([abs(pops[x]) for x in pops])/pop_old)
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if sum([abs(pops[x]) for x in pops]) >= pop_thresh and not vary_shift:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                S = utils.measure_hamil_overlap_groups_matrices(groups, mats, mcirc, mapping, shots)/utils.measure_overlap_shots_mat(mcirc, mapping, shots) - hf_e
                vary_shift = True
                step_0 = step
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if step%1 == 0: 
                f.write(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                print(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
            if step%50 == 0:
                print(step)
            t2 = time.time()
        f.write('Elapsed time: '+str(t2-t1)+ ' seconds')
        f.close()
    elif shots and gaussian_noise:
        epsilon = 1/shots
        for step in range(nsteps+1):
            
            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, noise_symbols, prob_noise, damp_prob) 
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params) 
            elif depolarising_noise or damping_noise:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 
                for d in params:
                    param_map.update(stoch_gates.get_noise(noise_symbols[d], noise_prob, damp_prob))
            else:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 

            current_ucc_circ = ucc_circ.copy()
            current_ucc_circ.symbol_substitution(param_map)
            pop_change = {}

            if stoch_hamil:
                groups_loc = get_stoch_hamil(keys, groups, nattempts, tot_weight, group_weights_cumul)

            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                if stoch_hamil:
                    ho = utils.measure_hamil_overlap_groups_matrices(groups_loc, mats, mcirc, mapping, epsilon)
                    overlap = utils.measure_overlap_shots_mat(mcirc, mapping, epsilon)
                    hamil_term = ho - (S + hf_e)*overlap
                else:
                    hamil_term = np.dot(mcirc.get_statevector(), np.dot(hmat, mcirc.get_statevector())).real + np.random.normal(0, epsilon)
                    hamil_term -= (S + hf_e) * (np.dot(mcirc.get_statevector(), np.dot(zmat, mcirc.get_statevector().T)).real + np.random.normal(0,epsilon))

                if det == ref:
                    e_proj = (np.dot(mcirc.get_statevector(), np.dot(hmat, mcirc.get_statevector().T)).real + np.random.normal(0,epsilon))/(np.dot(mcirc.get_statevector(), np.dot(zmat, mcirc.get_statevector().T)).real + np.random.normal(0,epsilon)) - hf_e
                pop_change[det] = -dt * pops[ref]*hamil_term*signs[det]

            for det in dets:
                pops[det] += pop_change[det]
            if vary_shift and (step - step_0) % 5 == 0:
                S -= xi/(5*dt)*np.log(sum([abs(pops[x]) for x in pops])/pop_old)
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if sum([abs(pops[x]) for x in pops]) >= pop_thresh and not vary_shift:
                
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                S = e_proj
                vary_shift = True
                step_0 = step
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if step%1 == 0: 
                f.write(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                print(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
        t2 = time.time()
        f.write('Elapsed time: '+str(t2-t1)+ ' seconds')
        f.close()
    else:
        for step in range(nsteps+1):
            
            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, noise_symbols, prob_noise, damp_prob) 
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params) 
            elif depolarising_noise or damping_noise:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 
                for d in params:
                    param_map.update(stoch_gates.get_noise(noise_symbols[d], noise_prob, damp_prob))
            else:
                param_map = {params[d]: pops[d]/pops[ref] for d in params} 

            current_ucc_circ = ucc_circ.copy()
            current_ucc_circ.symbol_substitution(param_map)
            pop_change = {}

            if stoch_hamil:
                groups_loc = get_stoch_hamil(keys, groups, nattempts, tot_weight, group_weights_cumul)
            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                if stoch_hamil:
                    ho = utils.measure_hamil_overlap_groups_matrices(groups_loc, mats, mcirc, mapping)
                    overlap = utils.measure_overlap_shots_mat(mcirc, mapping)
                    hamil_term = ho - (S + hf_e)*overlap
                else:
                    hamil_term = (mcirc.get_statevector()).dot(hmat.dot((mcirc.get_statevector()))).real
                    hamil_term -= (S + hf_e) * (mcirc.get_statevector()).dot(zmat.dot((mcirc.get_statevector()))).real

                if det == ref:
                    e_proj = (mcirc.get_statevector()).dot(hmat.dot((mcirc.get_statevector()))).real/(mcirc.get_statevector()).dot(zmat.dot((mcirc.get_statevector()))).real - hf_e
                pop_change[det] = -dt * pops[ref]*hamil_term*signs[det]
            for det in dets:
                pops[det] += pop_change[det]
            if vary_shift and (step - step_0) % 5 == 0:
                S -= xi/(5*dt)*np.log(sum([abs(pops[x]) for x in pops])/pop_old)
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if sum([abs(pops[x]) for x in pops]) >= pop_thresh and not vary_shift:
                
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits, current_ucc_circ)
                S = (mcirc.get_statevector()).dot(hmat.dot((mcirc.get_statevector()))).real/(mcirc.get_statevector()).dot(zmat.dot((mcirc.get_statevector()))).real - hf_e
                vary_shift = True
                step_0 = step
                pop_old = sum([abs(pops[x]) for x in pops])
        
            if step%1 == 0: 
                f.write(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
                print(str(step)+'    '+str (S)+'    '+str (e_proj)+'    '+str(pops[ref])+ '    ' +str(sum([abs(pops[x]) for x in pops]))+'    '+str({x:pops[x]/pops[ref] for x in pops})+'\n')
        t2 = time.time()
        f.write('Elapsed time: '+str(t2-t1)+ ' seconds')
        f.close()
