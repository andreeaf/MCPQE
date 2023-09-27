import numpy as np

a = np.zeros((5,5))

a[1-1,           1-1] =  -1.1889970389622466     
a[1-1,           2-1] =   1.5259225100150431E-007
a[1-1,           3-1] =  0.14623714015827219     
a[1-1,           4-1] =  -1.5259225100150431E-007
a[1-1,           5-1] =  0.13527783088942921     
a[2-1,           2-1] = -0.38698588239268927     
a[2-1,           3-1] =  0.14071950332879349     
a[2-1,           4-1] = -0.13527783088942921     
a[2-1,           5-1] =   1.3559859941521563E-002
a[3-1,           3-1] = -0.45321476790233217     
a[3-1,           4-1] = -0.14071950332879349     
a[3-1,           5-1] =  0.15892991286600680     
a[4-1,           4-1] = -0.38698588239268927     
a[4-1,           5-1] =  -1.3559859941521563E-002
a[5-1,           5-1] =  0.44976663900918534     

for i in range(5):
    for j in range(i+1, 5):
        a[j,i] = a[i,j]
ident = np.eye(5)
evals, evecs = np.linalg.eigh(a)
print(evals)
evals, evecs = np.linalg.eigh(a - a[0,0]*ident)
print(evals)

for omega in [0,0.25,0.5,0.75,1.0,1.25,1.50,1.75]:
    evals, evecs = np.linalg.eigh(np.matmul(a - (a[0,0]+omega)*ident, a - (a[0,0]+omega)*ident))
    hamil = np.matmul(a - (a[0,0]+omega)*ident, a - (a[0,0]+omega)*ident)
    f = open('hamil_omega='+str(omega)+'.dat', 'w')
    mapping = {0:3, 1:18, 2:12, 3:33, 4:48}
    for i in range(5):
        for j in range(i, 5):
            f.write(str(mapping[i]) + '    '+str(mapping[j])+'    '+str(hamil[i,j])+'\n')
    f.close()
    
        
    print(omega)
    print(evals)
    print(evecs[:,0])
    print()
    coeffs = np.array([1, 0, 0, 0, 0])
    ref_state = np.array([1, 0, 0, 0, 0])
    energy = np.dot(ref_state, np.dot( hamil,  coeffs.T)) / np.dot(ref_state,  coeffs.T)
    S = energy
    dt = 0.1
    print(energy, S, coeffs)
    vary_S = False
    for step  in range(1000):
        resid = np.dot((hamil - S * ident),  coeffs.T)
        coeffs = coeffs - dt *resid
        energy = np.dot(ref_state, np.dot( hamil,  coeffs.T)) / np.dot(ref_state,  coeffs.T)
        if vary_S and (step - step0)%5 == 0:
            S -= 0.1/(5*dt)*np.log(sum([abs(x) for x in coeffs])/pop_old) 
            pop_old = sum([abs(x) for x in coeffs])
        if sum([abs(x) for x in coeffs]) >= 2 and not vary_S:
            S = energy
            vary_S = True
            step0 = step
            pop_old = sum([abs(x) for x in coeffs])
    print(S,energy, [c/coeffs[0] for  c in coeffs])    
    print()
