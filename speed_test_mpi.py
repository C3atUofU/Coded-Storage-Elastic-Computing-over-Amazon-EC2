# -*- coding: utf-8 -*-

from mpi4py import MPI
import numpy as np
import copy
import time
import itertools
import multiprocessing


############
# MPI Set Up
############

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
N    = comm.Get_size()

    
################################
# Coded Elastic Computing Set Up
################################

np.random.seed(0)

T = 200

num_comp = 20

q = 20000

r_coded = 10000

#####################
# Generate Coded Data
#####################

A = np.random.rand(q,r_coded)

B = np.copy(np.transpose(A),order='C') # F vs. C makes a huge difference

b = np.random.rand(r_coded,1)

c = np.random.rand(q,1)

comps = np.zeros((T,num_comp+1),dtype='int32')

comps_2 = np.zeros((T,num_comp+1),dtype='int32')

for t in range(T):
    # Defining the computation splits of the data ahead of time
    # Row and column partitions
    comps_row = np.hstack((np.random.choice(q-1, num_comp-1, replace=False)+1,np.array([0,q])))
    comps_row.sort()
    comps[t,:] = comps_row
    comps_row = np.hstack((np.random.choice(r_coded-1, num_comp-1, replace=False)+1,np.array([0,r_coded])))
    comps_row.sort()
    comps_2[t,:] = comps_row

##################
# Run Computations
##################

time_vector = np.zeros((T,))

t_comp1 = 0

t_comp2 = 0

t1 = time.time()

for t in range(T):
    comps_coded = {}
    comps_coded_2 = {}
    t2 = time.time()
    for i in range(num_comp):
        #comps_coded_2[i] =  np.matmul(A[:,comps_2[t,i]:comps_2[t,i+1]].transpose(),c)  
        comps_coded_2[i] = np.matmul(B[comps_2[t,i]:comps_2[t,i+1],:],c) 
        #comps_coded_2[i] = np.matmul(np.copy(np.transpose(A[:,comps_2[t,i]:comps_2[t,i+1]]),order='C'),c)
        #comps_coded_2[i] = np.matmul(c, A[:,comps_2[t,i]:comps_2[t,i+1]])
    t_comp2 += 1000*(time.time()-t2)/T
    
    # comps_coded is for comp coded 1
    
    comps_coded = {}
    comps_coded_2 = {}
    t2 = time.time()
    for i in range(num_comp):
        comps_coded[i] = np.matmul(A[comps[t,i]:comps[t,i+1],:],b)
    t_comp1 += 1000*(time.time()-t2)/T
    
    time_vector[t] = time.time() - t1


comm.Barrier()

for i in range(N):
    time.sleep(0.01)
    if rank == i:
        print '-----------------------------'
        print 'Hi I am rank: ', rank
        print 'Processor   : ', MPI.Get_processor_name()    
        #print 'Comp 1      : %.2f' % t_comp1    
        #print 'Comp 2      : %.2f' % t_comp2            
        print 'Speed 1     : %.2f' % (1000/t_comp1)  
        print 'Speed 2     : %.2f' % (1000/t_comp2)   
        #print 'Speed Rat.  : %.2f' % (t_comp1/t_comp2)  
        print 'Speed       : %.2f' % (float(T)/time_vector[T-1]) 
        print 'Num. CPUs   : %.0f' % multiprocessing.cpu_count()
 
time_vector = comm.gather(time_vector,root=0)

if rank == 0:
    np.save('speed_run.npy',time_vector)