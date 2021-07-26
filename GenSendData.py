from mpi4py import MPI
from fractions import gcd
import numpy as np
import time
import copy
import os
import pickle
import sys

global_comm = MPI.COMM_WORLD
rank = global_comm.rank

global_comm.Barrier()
if rank == 0:
    timeCount = 0
    timeStamps = np.zeros((20,))
    timeStamps[timeCount] = time.time()
    print 'Master: COMM SET UP START'

# The below variables need to match the code that runs the coded terasort
# Match these items in: 'cdc_test_v19.py', 'cdc_uncoded_v1.py' etc...

dataDir = "data1"

total_data = 6 * 10 ** 8  # Number of key-value pairs to be sorted
num_files = 1000
key_max = 2 ** 16
value_max = 2 ** 16
vals_num_cols = 9
dataType = 'uint16'

array_len = int(total_data / num_files)
if rank == 0:  # Debugger
    print total_data
    print array_len


global_comm.Barrier()
if rank == 0:
    print 'Master: COMM SET UP COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]

if rank == 0:
    np.random.seed(0)

# Master Node: Generates Data and broadcasts to node placement groups
if not os.path.exists(dataDir):
    os.mkdir(dataDir)
elif os.path.exists(dataPath):
    os.remove(dataPath)

keys = []
values = []

for f in range(num_files):
    if rank == 0:
        # Master generates the files
        keys = np.random.randint(0, key_max, (array_len,), dtype=dataType)
        values = np.random.randint(0, value_max, (array_len, vals_num_cols), dtype=dataType)

    # The file are broadcast to the workers
    keys = global_comm.bcast(keys, root=0)
    values = global_comm.bcast(values, root=0)
    if rank != 0:
        # Workers save the files
        key_fname = dataDir + "/keys_%03d.npy" % f
        value_fname = dataDir + "/values_%03d.npy" % f
        np.save(key_fname,keys)
        np.save(value_fname,values)
    if rank == 0:
        print f

global_comm.Barrier()
if rank == 0:
    print 'Master: DATA GEN/SEND COMPLETE'
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print '\t Task time:', timeStamps[timeCount] - timeStamps[timeCount - 1]
#    print 'Total time:',timeStamps[timeCount]-timeStamps[0]
