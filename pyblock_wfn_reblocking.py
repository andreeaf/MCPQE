import numpy as np
import glob
import math
import pyblock
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Reblock QCQMC output files")

parser.add_argument('--save', type = str, help = "File to save results in")
parser.add_argument('--files', type = str, help = "Glob expression to match files against.")

args = parser.parse_args()

save = args.save
files = args.files

cols = [5*i + 9 for i in range(15)]
f = open(save, 'w')
for filename in glob.glob(files):
    d = np.genfromtxt(filename, usecols = tuple([1,2]+cols), unpack = True, skip_header=1, skip_footer=1)
    print(d[2], d[3], d[-1])
    for i in range(2000):
        if abs(d[0][i]) > 1e-10:
            starting_index = i+1
            break
    reblock_data = pyblock.blocking.reblock(d[:, starting_index:])
    #for reblock_iter in reblock_data:
    #    print(reblock_iter)

    opt = pyblock.blocking.find_optimal_block(len(d)-starting_index, reblock_data)
    print(filename)
    #print(opt)
    #print(reblock_data[opt[0]])
    f.write(str(filename))
    for i in range(2):
        f.write('   ' + str(reblock_data[opt[0]].mean[i]) + '    ' + str(reblock_data[opt[0]].std_err[i]))

    for i in range(2,17):
        f.write('   ' + str(reblock_data[opt[0]].mean[i]))
    f.write('\n')
f.close()

