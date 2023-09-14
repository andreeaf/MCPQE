import sym
sym.path.append('../')

import utils
import encode_ucc_wavefunction as euw

from sympy import Symbol
import numpy as np
from openfermion.transforms import jordan_wigner
def excit_to_det(occ_det, occ_ref, nqubits):
    det = sum(2**(nqubits-1-i) for i in occ_det)
    ref = sum(2**(nqubits-1-i) for i in occ_ref)
    #det = sum(2**i for i in occ_det)
    #ref = sum(2**i for i in occ_ref)
    excit = det^ref
    excit_level = utils.count_set_bits(excit)/2
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
dets = [(0,1), (2,3), (4,5), (0,5), (1,4)]
ref = (0,1)
#dets = [(0,1,2,3)]
nqubits = 6
dets.sort(key = lambda x: utils.dets_list_order(x, ref))
print(dets)
signs = {d:excit_to_det(d,ref, nqubits) for d in dets}
print(signs)
xi = 0.1
operators = utils.generate_excitation_operators(dets[1:], ref, nqubits)
qubit_ops = {op:jordan_wigner(operators[op]) for op in operators}
params = {d:Symbol('t'+str(utils.bits_shared_ref(d,ref,nqubits))+'_'+str(utils.bits_not_shared_ref(d,ref,nqubits))) for d in dets[1:]}
ucc_circ = euw.generate_ucc_state(ref, qubit_ops, params, nqubits)
pops = {d:np.random.rand() for d in dets[1:]}
pops[ref] = 1
param_map = {params[d]: pops[d]/pops[ref] for d in params}
param_map2 = {params[d]: np.arccos(-2*np.sin(pops[d]/pops[ref])+1)/2 for d in params}
#param_map2 = {params[d]: np.sqrt(pops[d]/pops[ref]) for d in params}
circ1 = ucc_circ.copy()
circ1.symbol_substitution(param_map)
circ2 = ucc_circ.copy()
circ2.symbol_substitution(param_map2)
print(param_map)
vector1 = circ1.get_statevector().real
norm = sum([i**2 for i in vector1])
print(norm)
#vector1 /= np.sqrt(norm)
vector1 /= norm
print(vector1)
print(param_map2)
vector2 = circ2.get_statevector().real
print(vector2)
print('diff',np.sum(abs(vector2 - vector1)))
coeff_signs = {d:np.sign(pops[d]) for d in pops}
print(coeff_signs)
vector2 = np.array([np.conj(vector2[i])*vector2[i] for i in range(2**6)]).real  
norm = sum([i**2 for i in vector2])
print(norm)
mapping = {(0,1):48, (2,3):12, (0,1):3, (0,5):33, (1,4):18}
for ref in mapping:
    vector2[mapping[ref]]*=coeff_signs[ref]*signs[ref]
vector2 /= np.sqrt(norm)

print(vector2)
print('diff',np.sum(abs(vector2 - vector1)))
