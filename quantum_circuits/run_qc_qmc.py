import hamil
import utils
import encode_ucc_wavefunction as euw

from openfermion.ops import FermionOperator
from openfermion.transforms import jordan_wigner

import numpy as np
import cProfile
import pstats

geometry = [['H', [0,0,0]], ['H', [0,0,1.0]]]
basis = 'sto-3g'
multiplicity = 1
charge = 0

qubit_h = hamil.get_qubit_molecular_hamiltonian(geometry, basis, multiplicity, charge)
ps_H = hamil.get_hamil_pauli_strings(qubit_h)
groups = utils.group_pauli_strings(ps_H)

pops = [1, 0]
dt = 0.5
dets = [[0,1], [2,3]]
ref = [0,1]
S = 0
step_0 = 0
pop_old = 1
xi = 1
vary_shift = False

operators = utils.generate_excitation_operators(dets[1:], ref, 4)
qubit_ops = [jordan_wigner(op) for op in operators]
#with cProfile.Profile() as profile:
if True:
    for step in range(1000):
    
        params = [pops[1]/pops[0]]
        ucc_circ = euw.generate_controlled_ucc_circ(4, qubit_ops, params)
        for det in dets:
            mcirc = euw.generate_controlled_overlap_measurement_circ(ref, det, 4, ucc_circ)
            hamil_term = pops[0]*(euw.measure_hamil_overlap(mcirc, groups, 4, True, 10000) -
                                  S * utils.measure_overlap(mcirc, True, 10000))
            #hamil_term2 = pops[0]*(euw.measure_hamil_overlap(mcirc, ps_H) -
            #                       S * utils.measure_overlap(mcirc))
            pops[dets.index(det)] -= dt*hamil_term
    
        if vary_shift and (step - step_0) % 10 == 0:
            S -= xi/(10*dt)*np.log(sum([abs(x) for x in pops])/pop_old)
            pop_old = sum([abs(x) for x in pops])
    
        if sum([abs(x) for x in pops]) >= 2 and not vary_shift:
            mcirc = euw.generate_controlled_overlap_measurement_circ(ref, ref, 4, ucc_circ)
            S = euw.measure_hamil_overlap(mcirc, groups, 4, True, 10000)
            vary_shift = True
            step_0 = step
            pop_old = sum([abs(x) for x in pops])
    
        if step%10 == 0: print(step, S, pops[0], pops[1]/pops[0])
#    ps = pstats.Stats(profile)
#    ps.sort_stats('cumtime')
#    ps.print_stats(10)
