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

f = open(save, 'w')
for filename in glob.glob(files):
    d = np.genfromtxt(filename, usecols = (1,2), unpack = True, skip_header=1, skip_footer=1)
    for i in range(len(d[0])):
        if abs(d[0][i]) > 1e-10:
            starting_index = i+1
            break
    reblock_data = pyblock.blocking.reblock(d[:, starting_index:])
    #for reblock_iter in reblock_data:
    #    print(reblock_iter)

    opt = pyblock.blocking.find_optimal_block(len(d), reblock_data)
    print(filename)
    #print(opt)
    #print(reblock_data[opt[0]])
    f.write(str(filename) + '   ' + str(reblock_data[opt[0]].mean[0]) + '    ' + str(reblock_data[opt[0]].std_err[0]) + '    '+ str(reblock_data[opt[0]].mean[1]) + '    '+ str(reblock_data[opt[0]].std_err[1]) + '    '+'\n')
f.close()
