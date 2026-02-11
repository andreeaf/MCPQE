'''Module for qc_qmc calculation'''

import copy
import time
import sys
from openfermion import jordan_wigner
from openfermion.linalg import qubit_operator_sparse
from sympy import Symbol
import numpy as np
import utils
import measurements
import encode_ucc_wavefunction as euw
import stoch_gates
import hamil
from hamil_decomps import sorted_insertion
from pytket.circuit import OpType

def get_stoch_wfn(pops, params, dets, ref, noise_symbols = None, prob = None, gamma = None):
    ''' Stochastically round amplitudes in the wavefunction.

    Args:
        pops: walker populations on excitors.
        params: circuit parameters.
        dets: list of determinants.
        ref: reference determinant.
        noise_symbols: symbols for noise channels.
        prob: noise probability
        gamma: damping probability.
    
    Return:
        param_map: truncated parameter map
    '''

    pops_loc = [(x, pops[x]) for x in dets[1:]]
    pops_loc.sort(key=lambda x: -abs(x[1]))
    if len(dets)>1:
        excit_pop = sum(abs(x) for x in [pops[x] for x in dets[1:]])
    else:
        excit_pop = 0
    rand = np.random.rand()*excit_pop
    param_map = {}
    cumul_pop = 0
    for term in pops_loc:
        if cumul_pop < rand:
            param_map[params[term[0]]] = term[1]/pops[ref]
            cumul_pop +=  abs(term[1])
            if noise_symbols is not None:
                param_map.update(stoch_gates.get_noise(noise_symbols[term[0]], prob, gamma))
        else:
            param_map[params[term[0]]] = 0
            if noise_symbols is not None:
                for symbol in noise_symbols[term[0]]:
                    param_map[symbol] = 0

    return param_map

def get_stoch_hamil(keys, groups, nattempts, tot_weight, group_weights_cumul, group_weights):
    ''' Stochastically sample the hamiltonian.
    
    Args:
        keys: hamiltonian group keys.
        groups: hamiltonian groups.
        nattempts: number of hamiltonian groups to sample.
        tot_weight: total weight of hamiltonian groups
        group_weights_cumul: cumulative weight of terms
        group_weights: weights of hamiltonian groups.

    Return:
        groups_loc: stochastically sampled hamiltonian groups.
    '''

    groups_loc = {keys[0]:groups[keys[0]]}
    for _ in range(nattempts):
        rand = np.random.rand()*tot_weight
        for grp in group_weights_cumul:
            if rand<group_weights_cumul[grp]:
                if grp in groups_loc:
                    for elem in range(len(groups_loc[grp])):
                        contrib = groups[grp][elem][1]/nattempts/group_weights[grp]*tot_weight
                        groups_loc[grp][elem][1] += contrib
                else:
                    groups_loc[grp] = copy.deepcopy(groups[grp])
                    for elem in groups_loc[grp]:
                        elem[1]*=tot_weight/(group_weights[grp])/nattempts
                break
    return groups_loc

def spawn(apply_mats, ucc_circ, param_map, dets, nqubits, ref, mapping, spawned_dets, pops):
    #TODO: update and write docs
    for _ , mat in apply_mats.items():
        circ_num = ucc_circ.copy()
        circ_num.symbol_substitution(param_map)
        counts = measurements.get_counts(np.dot(mat, circ_num.get_statevector()), 10, mapping)
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
        params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+
                    str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
        ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
        param_map = {params[d]: pops[d]/pops[ref] for d in params}
        signs = {d:utils.excit_to_det(d,ref) for d in dets}
    return operators, qubit_ops, params, ucc_circ, param_map, signs


def qc_qmc_ucc(ferm_hamil, ref, hf_e, nqubits, time_step, pop_thresh, filename, nsteps, coll_method,
               dets=None, stoch_hamil=False, sym=None, shots=False, total_shots=False, nattempts = 1,
               stoch_wfn=False, mats=None, apply_mats=None,
               depolarising_noise=False, noise_prob=0, gaussian_noise=False, damping_noise=False,
               damp_prob=0):
    '''Run MC-PQE calculation using the UCC ansatz.
    
    Args:
        ferm_hamil: Fermionic Hamiltonian operator.
        ref (tuple): occupation list for reference determinant.
        hf_e (float): Hartree-Fock energy.
        nqubits (int): number of qubits.
        time_step (float): size of QMC time-step.
        pop_thresh (int): population threshold.
        filename (str): output file.
        nsteps (int): number of QMC steps.
        coll_method (str): Hamiltonian collection method (QWC or SI)
        dets (list): occupation list for determinants in UCC ansatz.
        stoch_hamil (bool): if True sample the Hamiltonian.
        sym (bool): if True
        shots (int): number of shots used to measure observables per collection. 
                     If 0 (False), carry out statevector simulation.
        total_shots (int): total number of shots used to measure observables.
        nattemps (int): number of hamiltonian terms to sample
        stoch_wfn (bool): if True, stochastically truncate wavefunction.
        hmat (str):
        zmat (str):
        mats (str):
        apply_mats (str):
        depolarising_noise (bool): if True, apply depolarising noise after every UCC excitation operation
        noise_prob (float): probability of depolarising noise
        gaussian_noise (bool): if True, use gaussian noise instead of true shots
        damping noise (bool): if True, apply damping noise after every UCC excitation operation
        damp_prob (float): damping noise probability
    '''
    QWC: bool
    if coll_method.lower() == "qwc":
        QWC = True
        print("QWC")
    else:
        QWC = False
        print("SI")

    Z = np.array([[1,0],[0,-1]])

    #Map integers to occupation strings of the whole system + ancilla Hilbert space
    mapping = {det: utils.get_bits(det, nqubits +1) for det in range(2**(nqubits+1))}
    time1 = time.time()

    #Decide if spawning is necessary
    spawning = False
    if dets is None:
        spawning = True
        dets = [ref]
    if damp_prob:
        damp_prob = 2 * np.arcsin(np.sqrt(damp_prob))/np.pi

    #Transform Hamiltonian into qubit basis and group
    qubit_h = jordan_wigner(ferm_hamil)
    ps_h = hamil.get_hamil_pauli_strings(qubit_h)
    groups = utils.group_pauli_strings(ps_h)
    collections = sorted_insertion(fermi_hamil=ferm_hamil)

    dets.sort(key = lambda x: utils.dets_list_order(x, ref))

    #Get excitation operations corresponding to determinants
    operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
    qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}

    #Get symbolic UCC parameters and create symbolic circuit
    params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+
                str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
    ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)

    if mats is None:
        if QWC:
            mats = hamil.get_hamil_group_matrices(groups, nqubits, filename+'_mats')

        else:
            mats = hamil.get_hamil_si_collection_msmt_matrices(collections, nqubits)
            mats_with_coeffs = [
                (mat[0], mat[1], mat[2], collections[i])
                for i, mat in enumerate(mats)
            ]  # list of (collection msmt matrix, qubits to measure, phases, collection) for each collection
            sq_mats_with_coeffs = hamil.get_hamil_sq_si_collection_msmt_matrices(collections, nqubits)  #list of (msmt circuit matrix, qubits to measure, phases, coeffs, identity term) for each collection

    num_colls = len(mats)
    if total_shots and not shots:
        # shots used to get initial measurement split
        shots = int(np.ceil(total_shots/num_colls))

    #Generate probabilities for Hamiltonian term sampling
    if stoch_hamil:
        group_weights = {key:sum(abs(group[i][1]) for i in
                            range(len(group))) for key, group in groups.items()}
        keys = list(groups.keys())
        group_weights_cumul = {}
        for i in range(1, len(keys)):
            group_weights_cumul[keys[i]] = sum(group_weights[keys[i]] for i in range(1,i+1))
        tot_weight = sum(group_weights[keys[i]] for i in range(1,i+1))

    #Sort determinants according to "standard" order
    dets.sort(key = lambda x: utils.dets_list_order(x, ref))
    zeta = 1 #TODO: Pass in as parameter

    #Generate excitation operators corresponding to determinants
    operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
    qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
    params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+
                str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}

    #Generate corresponding UCC ansatz and all measurement circuits
    if depolarising_noise or damping_noise:
        noise_symbols = {}
        ucc_circ = stoch_gates.generate_noisy_controlled_ucc_circ(nqubits, qubit_ops,
                                params, noise_symbols, depolarising_noise = depolarising_noise,
                                amplitude_damping = damping_noise)
    else:
        ucc_circ = euw.generate_controlled_ucc_circ(nqubits, qubit_ops, params)
    if spawning:
        #TODO: bring spawning up to date.
        #Generate spawning circuits.
        ucc_circ2 = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
        if apply_mats is None:
            apply_mats = hamil.get_hamil_matrix(ps_h, nqubits)
        spawned_dets = []

    #Initialise QMC
    signs = {d:utils.excit_to_det(d,ref) for d in dets}
    pops = {d:0 for d in dets[1:]}
    pops[ref] = 1
    shift = 0
    step_0 = 0
    pop_old = 1
    vary_shift = False


    #TODO: More rigorous trial wfns.
    #trial_wfn = {ref:1.0}
    #, (2,3): -0.5, (4,5):-0.35}
    #dets_trial = {(2,3), (4,5)}
    #Generate excitation operators corresponding to trial determinants
    #operators_trial = utils.generate_excitation_operators(dets_trial, ref, nqubits)
    #qubit_ops_trial = {op:jordan_wigner(operators[op]) for op in operators_trial}
    #params_trial = {d: trial_wfn[d] for d in dets_trial}
    #ucc_circ_trial = euw.generate_controlled_ucc_circ(nqubits, qubit_ops_trial, params_trial)

    
    #Open output file
    output_file = open(filename, 'a', encoding="utf-8")
    output_file.write('Starting QMC \n')
    output_file.write(f"Collection method: {coll_method}\n")
    output_file.write('step      S         E_proj        N0         N_tot       Ni/N0\n')
    e_proj = 0

    #Do QMC
    if (not shots) and gaussian_noise:
        print('Invalid parameters. Asked for Gaussian noise but no number of shots provided.')
        sys.exit()
    if shots and not gaussian_noise:
        e_proj_list = []
        msmt_split = []
        shot_split_by_det = {}
        NUM_DETS = len(dets)
        for step in range(nsteps+1):
            if step%10 == 0:
                print(f"STEP {step}")
            e_proj = 0
            ov_proj = 0
            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, dets, ref,
                                          noise_symbols, noise_prob, damp_prob)
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params, dets, ref)
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
                groups_loc = get_stoch_hamil(keys, groups, nattempts,
                                             tot_weight, group_weights_cumul, group_weights)

            # MEASUREMENT SPLITTING METHODS

            # 1. HARDCODED
            # msmt_split = [20000, 30000]


            # 2, 3. ALL DETS, INDIVIDUAL DET SHOT SPLIT OR RESVAR SUM OVER COLLECTIONS WEIGHTED BY |RESIDUAL|
            if step == 0 and (not QWC) and (not stoch_hamil) and total_shots:
                
                # 2. DIFFERENT SHOT SPLIT FOR EACH DET (NO SUMMING RESVAR)
                # for det in dets:
                #     mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                #                                                         nqubits, current_ucc_circ)
                #     det_resvars = measurements.get_resvars(nqubits, sq_mats_with_coeffs, mats_with_coeffs, mcirc, mapping, shots, gaussian_noise)
                #     det_resvars = np.array(det_resvars)
                #     det_resvars = np.abs(det_resvars) # !! FIND A BETTER SOLUTION TO NEGATIVE RESVARS !!
                #     det_resstds = np.sqrt(det_resvars)
                #     shot_split = np.ceil(total_shots * det_resstds / np.sum(det_resstds)).astype(int).tolist()
                #     shot_split_by_det[det] = shot_split

                # print(shot_split_by_det)


                # 3. SAME SHOT SPLIT FOR EACH DET (SUM RESVARS ACROSS DETS)
                # resvars = []
                # for det in dets:
                #     if det != ref:  
                #         mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                #                                                             nqubits, current_ucc_circ)
                #         det_resvars = measurements.get_resvars(nqubits, sq_mats_with_coeffs, mats_with_coeffs, mcirc, mapping, shots, gaussian_noise)
                #         # if not np.all(np.round(det_resvars, 9) > 0):
                #         #     print(det)
                #         #     statevec = mcirc.get_statevector()
                #         #     with open(f"{det}_starting_statevec.npy", "wb") as f:
                #         #         np.save(f, statevec)
                #         #     with open(f"{det}_starting_resvar.npy", "wb") as f:
                #         #         np.save(f, det_resvars)
                #         resvars.append(np.abs(det_resvars).tolist())
                # resvar_sum = np.sum(np.array(resvars), axis=0)
                # resstd_sum = np.sqrt(resvar_sum)  # try split by residual stdev instead of resvar
                # msmt_split = np.ceil(total_shots * resstd_sum / np.sum(resstd_sum)).astype(int).tolist()
                # init_msmt_split = msmt_split
                # print(init_msmt_split)

                # # residual-weighted sum of resvars across dets
                # init_resids = []  
                # for det in dets:
                #     if det != ref:
                #         mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                #                                                                 nqubits, current_ucc_circ)
                #         statevec = mcirc.get_statevector()
                #         hamil_overlap = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, statevec, mapping, shots, shot_split=init_msmt_split)
                #         overlap = measurements.measure_overlap_shots_mat(statevec, mapping, shots)
                #         resid = pops[ref]*(hamil_overlap - (shift + hf_e) * overlap)
                #         init_resids.append(np.abs(resid))

                # init_resid_weights = np.array(init_resids) / np.sum(init_resids)
                # init_weighted_resvars = [weight*np.array(resvars[idx]) for idx, weight in enumerate(init_resid_weights)]
                # init_weighted_resvar_sum = np.sum(np.array(init_weighted_resvars), axis=0)
                # init_weighted_resstd_sum = np.sqrt(init_weighted_resvar_sum)
                # weights = (init_weighted_resstd_sum / np.sum(init_weighted_resstd_sum))  # standard weighted, more shots for larger resvar
                # # weights = (1 - init_weighted_resvar_sum / np.sum(init_weighted_resvar_sum)) / (len(init_weighted_resvar_sum) - 1)  # reverse weighted, fewer shots for larger resvar
                # # print(weights)
                # msmt_split = np.ceil(total_shots * weights).astype(int).tolist()
                # print(msmt_split)


                # 4. EACH DET HAS DIFFERENT TOTAL SHOTS, DIFFERENT SHOT DISTRIBUTION
                collection_mats_with_ancilla = []
                for collection in collections:
                    d_mat = np.zeros(shape=(2**nqubits, 2**nqubits))
                    for op in collection:
                        d_mat = d_mat + qubit_operator_sparse(op, nqubits).toarray()
                    zd_mat = np.kron(Z, d_mat)
                    collection_mats_with_ancilla.append(zd_mat)

                total_shots_all_dets = NUM_DETS*total_shots
                resvars_by_det_and_coll = []  # [[det 1: coll 1, ..., coll n], [det 2: coll 1, ..., coll n], ...]
                for det in dets:
                    mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                                                                        nqubits, current_ucc_circ)
                    curr_statevec = mcirc.get_statevector()
                    det_resvar_list = []
                    for zd_mat in collection_mats_with_ancilla:
                        sq_term = np.real(curr_statevec @ (zd_mat @ zd_mat) @ curr_statevec.T)
                        term = np.real(curr_statevec @ zd_mat @ curr_statevec)
                        det_resvar_list.append(np.round(sq_term - term**2, 9))
                    resvars_by_det_and_coll.append(det_resvar_list)
                
                if not np.all(np.array(resvars_by_det_and_coll) >= 0):
                    print("SERIOUS ERROR - EXACTLY COMPUTED RESVAR IS NEGATIVE AT THESE POSITIONS ([DET IDX, COLL IDX]): ")
                    print(np.argwhere(np.array(resvars_by_det_and_coll) < 0))
                resvar_sum_over_colls = np.sum(resvars_by_det_and_coll, axis=1)  # sum of resvar across all colls for each det
                resstd_over_colls = np.sqrt(resvar_sum_over_colls)
                det_total_shot_split = np.ceil(total_shots_all_dets * resstd_over_colls / np.sum(resstd_over_colls)).astype(int).tolist()

                for det_idx, det in enumerate(dets):
                    det_total_shots = det_total_shot_split[det_idx]
                    det_resstd_list = np.sqrt(resvars_by_det_and_coll[det_idx])
                    det_shot_split = np.ceil(det_total_shots * det_resstd_list / np.sum(det_resstd_list)).astype(int).tolist()
                    shot_split_by_det[det] = det_shot_split

                
            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                                                                    nqubits, current_ucc_circ)
                statevec = mcirc.get_statevector()
                if stoch_hamil:
                    hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups_loc,
                                                                    mats, mcirc, mapping, shots)
                else:
                    if QWC:
                        hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups, mats,
                                                                        mcirc, mapping, shots)
                    else:
                        if total_shots:
                            msmt_split = shot_split_by_det[det] # FOR METHODS 2 OR 4 ABOVE
                            hamil_overlap = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, statevec, mapping, shots, shot_split=msmt_split)
                        else:
                            hamil_overlap = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, statevec, mapping, nshots=shots)
                overlap = measurements.measure_overlap_shots_mat(statevec, mapping, shots)
                hamil_term = pops[ref]*(hamil_overlap - (shift + hf_e) * overlap)
                if det == ref:
                    e_proj = (hamil_overlap/overlap - hf_e)
                pop_change[det] = -time_step* hamil_term*signs[det]
                #if det in trial_wfn:
                #    e_proj += trial_wfn[det]*hamil_overlap
                #    ov_proj+= trial_wfn[det]*overlap


            for det in dets:
                pops[det] += pop_change[det]
            if spawning:
                operators, qubit_ops, params, ucc_circ, param_map, signs = spawn(apply_mats,
                                                    ucc_circ2, param_map, dets, nqubits, ref,
                                                    mapping, spawned_dets, pops)

            if vary_shift and (step - step_0) % 5 == 0:
                shift -= zeta/(5*time_step)*np.log(sum(abs(pops[x]) for x in pops)/pop_old)
                pop_old = sum(abs(pops[x]) for x in pops)

            #Compute shift the first time population crosses threshold.
            if sum(abs(pops[x]) for x in pops) >= pop_thresh and not vary_shift:

                # MEASUREMENT SPLIT FIRST TIME SHIFT IS NONZERO
                # if (not QWC) and (not stoch_hamil) and total_shots:
                #     new_msmt_split = []
                #     new_resids = []
                #     resvars = []
                #     for det in dets:
                #         mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, nqubits, current_ucc_circ)
                #         det_resvars = measurements.get_resvars(nqubits, sq_mats_with_coeffs, mats_with_coeffs, mcirc, mapping, shots, gaussian_noise, msmt_split)
                #         resvars.append(det_resvars)

                #         hamil_overlap = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, mcirc, mapping, shots, shot_split=init_msmt_split)
                #         overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, shots)
                #         resid = pops[ref]*(hamil_overlap - (shift + hf_e) * overlap)
                #         new_resids.append(np.abs(resid))

                #     new_resid_weights = np.array(new_resids) / np.sum(new_resids)
                #     new_weighted_resvars = [weight*np.array(resvars[idx]) for idx, weight in enumerate(new_resid_weights)]
                #     new_weighted_resvar_sum = np.sum(np.array(new_weighted_resvars), axis=0)
                #     # weights = new_weighted_resvar_sum / np.sum(new_weighted_resvar_sum)
                #     weights = 1 - new_weighted_resvar_sum / np.sum(new_weighted_resvar_sum)
                #     new_msmt_split = np.ceil(total_shots * weights).astype(int).tolist()
                #     msmt_split = new_msmt_split
                #     print(new_msmt_split)

                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, nqubits,
                                                                          current_ucc_circ)
                statevec = mcirc.get_statevector()
                if QWC:
                    shift = measurements.measure_hamil_overlap_groups_matrices(groups, mats, mcirc,
                                    mapping, shots)/measurements.measure_overlap_shots_mat(mcirc,
                                                        mapping, shots) - hf_e
                else:
                    if total_shots:
                        msmt_split = shot_split_by_det[ref]  # FOR METHODS 2 OR 4 ABOVE
                        shift = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, statevec, mapping, shots, 
                                                                        shot_split=msmt_split)/measurements.measure_overlap_shots_mat(statevec, mapping, shots) - hf_e
                    else:
                        shift = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, statevec, mapping, 
                                                                        shots)/measurements.measure_overlap_shots_mat(statevec, mapping, shots) - hf_e
                vary_shift = True
                step_0 = step
                pop_old = sum(abs(pops[x]) for x in pops)
            e_proj_list.append(e_proj)
            if step%10 == 0:
                output_file.write(str(step)+'    '+str (shift)+'    '+str (e_proj)+'    '+
                        str(pops[ref])+'    ' +str(sum(abs(pops[x]) for x in pops))+'    '+
                        str({x:pops[x]/pops[ref] for x in pops})+'\n')
            time2 = time.time()
        output_file.write('Elapsed time: '+str(time2-time1)+ ' seconds')
        output_file.close()
        with open("e_proj_list.npy", "wb") as f:
            np.save(f, e_proj_list)
    elif shots and gaussian_noise:
        epsilon = 1/shots
        for step in range(nsteps+1):

            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, dets, ref,
                                          noise_symbols, noise_prob, damp_prob)
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params, dets, ref)
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
                groups_loc = get_stoch_hamil(keys, groups, nattempts, tot_weight,
                                             group_weights_cumul, group_weights)

            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                                                        nqubits, current_ucc_circ)
                if stoch_hamil:
                    hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups_loc,
                                                         mats, mcirc, mapping, shots, gaussian_noise)
                    overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, shots, gaussian_noise)
                    hamil_term = hamil_overlap - (shift + hf_e)*overlap
                else:
                    hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups,
                                                         mats, mcirc, mapping, shots, gaussian_noise)
                    overlap = measurements.measure_overlap_shots_mat(mcirc, mapping, shots, gaussian_noise)
                    hamil_term = hamil_overlap - (shift + hf_e)*overlap


                if det == ref:
                     e_proj = hamil_overlap/overlap - hf_e
                pop_change[det] = -time_step * pops[ref]*hamil_term*signs[det]

            for det in dets:
                pops[det] += pop_change[det]
            if vary_shift and (step - step_0) % 5 == 0:
                shift -= zeta/(5*time_step)*np.log(sum(abs(pops[x]) for x in pops)/pop_old)
                pop_old = sum(abs(pops[x]) for x in pops)

            #Compute shift the first time population crosses threshold.
            if sum(abs(pops[x]) for x in pops) >= pop_thresh and not vary_shift:

                shift = e_proj
                vary_shift = True
                step_0 = step
                pop_old = sum(abs(pops[x]) for x in pops)

            if step%10 == 0:
                output_file.write(str(step)+'    '+str (shift)+'    '+str (e_proj)+'    '+
                        str(pops[ref])+'    ' +str(sum(abs(pops[x]) for x in pops))+'    '+
                        str({x:pops[x]/pops[ref] for x in pops})+'\n')
        time2 = time.time()
        output_file.write('Elapsed time: '+str(time2-time1)+ ' seconds')
        output_file.close()
    else:
        for step in range(nsteps+1):
            if step%10 == 0:
                print(f"STEP {step}")
            if stoch_wfn and (depolarising_noise or damping_noise):
                param_map = get_stoch_wfn(pops, params, dets, ref,
                                          noise_symbols, noise_prob, damp_prob)
            elif stoch_wfn:
                param_map = get_stoch_wfn(pops, params, dets, ref)
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
                groups_loc = get_stoch_hamil(keys, groups, nattempts, tot_weight,
                                            group_weights_cumul, group_weights)
            for det in dets:
                mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det,
                                                        nqubits, current_ucc_circ)
                if stoch_hamil:
                    hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups_loc,
                                                                  mats, mcirc, mapping)
                    overlap = measurements.measure_overlap_shots_mat(mcirc, mapping)
                    hamil_term = hamil_overlap - (shift + hf_e)*overlap
                else:
                    hamil_overlap = 0
                    if QWC:
                        hamil_overlap = measurements.measure_hamil_overlap_groups_matrices(groups,
                                                                  mats, mcirc, mapping)  # <phi_i | H | psi(beta)>
                    else:
                        hamil_overlap = measurements.measure_collection_overlap(nqubits, mats_with_coeffs, mcirc, mapping)
                    overlap = measurements.measure_overlap_shots_mat(mcirc, mapping)  # <phi_i | psi(beta)>
                    hamil_term = hamil_overlap - (shift + hf_e)*overlap  # residual r_i
                if det == ref:
                     e_proj = hamil_overlap/overlap - hf_e
                pop_change[det] = -time_step * pops[ref]*hamil_term*signs[det]
            for det in dets:
                pops[det] += pop_change[det]
            if vary_shift and (step - step_0) % 5 == 0:
                shift -= zeta/(5*time_step)*np.log(sum(abs(pops[x]) for x in pops)/pop_old)
                pop_old = sum(abs(pops[x]) for x in pops)

            #Compute shift the first time population crosses threshold.
            if sum(abs(pops[x]) for x in pops) >= pop_thresh and not vary_shift:

                shift = e_proj
                vary_shift = True
                step_0 = step
                pop_old = sum(abs(pops[x]) for x in pops)

            if step%10 == 0:
                output_file.write(str(step)+'    '+str (shift)+'    '+str (e_proj)+'    '+
                        str(pops[ref])+'    ' +str(sum(abs(pops[x]) for x in pops))+'    '+
                        str({x:pops[x]/pops[ref] for x in pops})+'\n')
        time2 = time.time()
        output_file.write('Elapsed time: '+str(time2-time1)+ ' seconds')
        output_file.close()
