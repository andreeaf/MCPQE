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
    print(omega)
    print(evals)
    print(evecs[:,0])
    print(np.sqrt(evals[0]) + omega)
    print(-np.sqrt(evals[0]) + omega)
    print()
