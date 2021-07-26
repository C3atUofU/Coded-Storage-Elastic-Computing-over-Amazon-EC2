# -*- coding: utf-8 -*-

# v01 of this code altered from v14 CEC_test

from mpi4py import MPI
import numpy as np
import copy
import time
import itertools

############
# MPI Set Up
############

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

T = 300

if rank == 0:
    timeCount = 0
    timeStamps = np.zeros(((T+2)*10,))
    timeTrack  = np.zeros((10,))
    timeStamps[timeCount] = time.time()
    print 'Master: SET UP START'  
    run_file = open("run_info.txt","w") 
    
################################
# Coded Elastic Computing Set Up
################################

np.random.seed(0)

Nmax = comm.Get_size() - 1

option = 'HetAlgorithm'

#option = 'HomCyclic'

#option = 'Uncoded'

L = 10

Str_Tol = 2

K = L + Str_Tol

q_coded = 20000

r_coded = 500

q = q_coded*L

r = r_coded*L

eta = 0.0000001  # training rate

gamma = 0.000001 # regulatization coefficient

speed_update_factor = 0.5

if rank == 0:
    print option
    print Str_Tol
    run_file.write(option)
    run_file.write('%d \n' %Str_Tol)

speed_comp = np.hstack((1.0*np.ones((L,)),1.0*np.ones((Nmax-L,))))

p_available = np.hstack((np.array([1 for n in range(K)]),np.array([0.5 for n in range(Nmax-K)])))

# Memory allocation (speeds up first time step)
comp_assignment = {'machines': [frozenset([int(n+1) for n in range(K)])],
                   'row_sets': [0,r_coded],
                   'w': []}
comp_assignment['machines2fail'] = set([])

comp_assignment['w']= np.random.rand(r,1) 
            
comp_assignment = comm.bcast(comp_assignment,root=0)

comp_coded = {}

available_machines = np.linspace(1,L,L).astype(int)

for i in range(Nmax):
    comp_coded[i] = 10*np.random.rand(int(L*q_coded/Nmax),1)
        
comp_coded = comm.gather(comp_coded,root=0)

###########################
# Generate Code for Storage
###########################

G = np.identity(L)

G_rand_gen =  np.array([[   0.8040,    0.5490,    0.8470,    0.2480,    0.4620,    0.0460,    0.3410,    0.7130,    0.8330,    0.2930],
     [0.0060,    0.4840,    0.1450,    0.6400,    0.2560,    0.5430,    0.0780,    0.1070,    0.1740,    0.9260],
     [0.1340,    0.0800,    0.8780,    0.8180,    0.7260,    0.5680,    0.2040,    0.1080,    0.4140,    0.4250],
     [0.3150,    0.4430,    0.0430,    0.9820,    0.7290,    0.0430,    0.6500,    0.6790,    0.6380,    0.3060],
     [0.8450,    0.0150,    0.3120,    0.1750,    0.5040,    0.0080,    0.5400,    0.9570,    0.7490,    0.7280],
     [0.4980,    0.5450,    0.1480,    0.3950,    0.9650,    0.2660,    0.1340,    0.7110,    0.8380,    0.5720],
     [0.4250,    0.5850,    0.9620,    0.2380,    0.8990,    0.6980,    0.7050,    0.5740,    0.2350,    0.0340],
     [0.8180,    0.1660,    0.0870,    0.4850,    0.3200,    0.6930,    0.2430,    0.1510,    0.1050,    0.0980],
     [0.7010,    0.2450,    0.4630,    0.7500,    0.0090,    0.9160,    0.6800,    0.1140,    0.9910,    0.0390],
     [0.0800,    0.2630,    0.1380,    0.6720,    0.6310,    0.3870,    0.3230,    0.9810,    0.0690,    0.0540]])

G = np.vstack((G,G_rand_gen))


if rank == 0:
    H_code = {}
    if option != 'Uncoded':
        decode_sets = [frozenset(a) for a in list(itertools.combinations(range(1,Nmax+1),L))]
        for dec_set in decode_sets:
            dec_list = list(dec_set)
            dec_list.sort()
            machine_set = np.array(dec_list).astype(int)
            H = np.linalg.inv(G[machine_set-1,:])  
            H_code[dec_set] = {}
            for i in range(L):
                H_code[dec_set][machine_set[i]] = H[:,i]  
    else:
        decode_sets = [frozenset((range(1,L+1)))]
        H = np.eye(L)
        H_code[decode_sets[0]] = {}
        for i in range(L):
            H_code[decode_sets[0]][i+1] = H[i,:]

    
if rank == 0:
    timeCount += 1
    timeStamps[timeCount] = time.time()
    print 'Master: SET UP COMPLETE'   
    print '\t Task time:',timeStamps[timeCount]-timeStamps[timeCount-1]

#####################
# Generate Coded Data
#####################

if rank == 0:
    np.random.seed(1)
    Err = np.zeros((T,))
    iteration_time = np.zeros((T,))
    x_real = np.random.rand(r,1)
    x = np.random.rand(r,1)
    z = np.zeros((q,1))
    y = 0.05*np.random.randn(q,1)
    
else:
    A_coded = np.zeros((q_coded,r))
    A_coded_T = np.zeros((r_coded,q))

np.random.seed(0)

for i in range(L):
    for j in range(L):
        A = np.random.rand(q_coded,r_coded)*float(100)/float(r*q)
        if j == L-1:
            A[:,r_coded-1] = 1
        if rank == 0:
            y[i*q_coded:(i+1)*q_coded] += np.matmul(A,x_real[j*r_coded:(j+1)*r_coded])
        else:
            A_coded[:,j*r_coded:(j+1)*r_coded]   += G[rank-1,i] * A
            A_coded_T[:,i*q_coded:(i+1)*q_coded] += G[rank-1,j] * A.T

comm.Barrier()
if rank == 0:
    timeCount +=1
    timeStamps[timeCount] = time.time()
    print 'Master: GEN DATA COMPLETE'    
    print '\t Task time:',timeStamps[timeCount]-timeStamps[timeCount-1]   

##################
# Run Computations
##################


for t in range(T):

    #######################
    # Computation Time Step
    #######################
    
    if rank == 0:  
        # Master node

        comp_assignment['w'] = np.copy(x)
        #comp_assignment['w'] = np.ones((r,1))
        
        if option != 'Uncoded':         
            available_machines = np.where(p_available >= np.random.rand(1,Nmax))[1]+1
            Nt = len(available_machines)
            num_fail = Str_Tol
            if Str_Tol > 0:
                # VVV STRAGGLER MODEL 2 VVV
                #comp_assignment['machines2fail'] = set(np.random.choice(available_machines,num_fail,replace=False))
                # VVV STRAGGLER MODEL 1 VVV
                comp_assignment['machines2fail'] = set(available_machines[np.argsort(speed_comp[available_machines-1])[:Str_Tol]])
        
        if option == 'Uncoded':
            comp_assignment['row_sets'] = [0, q_coded]
            
        if option == 'HetAlgorithm':            
            
            #########################
            # Define Computation Load
            #########################
            
            #print '--------------'
            speed_aiv = copy.copy(speed_comp)
            for i in range(1,Nmax+1):
                if i not in available_machines:
                    speed_aiv[i-1] = 0
            #print speed_aiv        
            comp_load = np.zeros((Nmax,))
            
            if Nt == K:
                comp_load = float(1)*(speed_aiv>0)
            else:
                ind_sort = np.argsort(speed_aiv)
                speed_aiv = speed_aiv[ind_sort]
                ind_unsort = np.argsort(ind_sort)
                k = Nmax

                while np.sum(speed_aiv[0:k]) < speed_aiv[k-1]*(K - Nmax + k):
                    k = k - 1
                    
                c = float(K - Nmax + k)/(np.sum(speed_aiv[0:k]))
                comp_load[0:k] = c*speed_aiv[0:k]
                comp_load[k:Nmax] = 1
                comp_load = comp_load[ind_unsort]
                
            
            #print comp_load
            #print '-  -  -  -  - '
            
            ########################################################
            # Define Specific Computation Assignment using Algorithm
            ########################################################
        
            comp_assignment['machines'] = []
        
            comp_assignment['row_sets'] = [0]    
                        
     
                
            m = np.copy(comp_load)
            
            alpha_sum = float(0)
            
            alpha = np.zeros((Nmax,))
            
            iteration = 0
            
            error_tol = 1e-4
            
            while alpha_sum < (1-error_tol) and iteration < Nmax:
                Np = np.sum(m>0)
                machine_argsort = m.argsort().astype(int)
                machine_smallest = machine_argsort[-Np]
                machines_largest = machine_argsort[-(K-1):Nmax]
                
                if  Np > K:
                    machine_Kth_largest = machine_argsort[-K]
                    alpha[iteration] = np.min(
                                (np.sum(m)/float(K)-m[machine_Kth_largest],
                                 m[machine_smallest]))    
                    
                else:
                    alpha[iteration] = m[machine_smallest]
                    
                machine_set = [machine_smallest + 1]
                machine_set.extend(list(machines_largest + 1))
                
                comp_assignment['machines'].append(frozenset(machine_set))
                
                m[machine_smallest] -= alpha[iteration]
                m[machines_largest] -= alpha[iteration]
                m[m < error_tol] = 0
                alpha_sum += alpha[iteration]
                iteration += 1
                            
            alpha = np.cumsum(alpha)
            alpha[iteration-1] = 1
            for i in range(iteration):
                comp_assignment['row_sets'].append(int(alpha[i]*q_coded))   
            
        if option == 'HomCyclic':
            
            #######################
            # Scheme of Yang et al.
            #######################
            comp_load = np.array([float(K) / float(Nt) for n in range(Nmax)])
            comp_assignment['row_sets'] = list(np.linspace(0,q_coded,Nt+1).astype(int))
            b =  np.linspace(0,K-1,K).astype(int) # np.random.permutation(L) 
            comp_assignment['machines'] = \
                    [frozenset(available_machines[(b+n)%Nt]) for n in range(Nt)] 
            
        #print comp_assignment['row_sets']
        #print comp_assignment['machines']
                      
    comm.Barrier()
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[0] += timeStamps[timeCount]-timeStamps[timeCount-1] 
       
    ########################################
    # Send Computation Assignment to Workers
    ########################################
       
    comp_assignment = comm.bcast(comp_assignment,root=0)
        
    if rank == 0:
        # Master node
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[1] += timeStamps[timeCount]-timeStamps[timeCount-1]  
        if t > 0:
            if t % 10 == 0:
                print t        
                run_file.write('%d \n' %t)
        
    #############################################
    # Workers Perform Assigned Coded Computations
    #############################################
    
    comp_coded = {}
    
    t1 = time.time()
    if rank not in comp_assignment['machines2fail']:
        for i in range(len(comp_assignment['machines'])):
            dec_set = comp_assignment['machines'][i]
            if rank in dec_set and rank > 0:
                comp_coded[i] = np.matmul(
                                    A_coded[comp_assignment['row_sets'][i]:comp_assignment['row_sets'][i+1]],
                                    comp_assignment['w'])
    
    t2 = time.time() - t1
        
    comm.Barrier()
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[2] += timeStamps[timeCount]-timeStamps[timeCount-1] 
    
    ##################################################
    # Master Gathers the Computations from the Workers
    ##################################################
    
    comp_coded = comm.gather(comp_coded,root=0)
    
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[3] += timeStamps[timeCount]-timeStamps[timeCount-1] 
    
    ########################################################
    # Master Combines Worker Results for Intermediate Result
    ########################################################
    
    if rank == 0:
        # Master node
        z = np.copy(-y)
        for i in range(len(comp_assignment['machines'])):
            try:
                machine_set = frozenset(np.random.choice(
                        list(comp_assignment['machines'][i] - comp_assignment['machines2fail']),
                        L,replace=False))
            except:
                print 'FAIL FAIL FAIL FAIL FAIL'
                print comp_assignment['machines'][i]
                print comp_assignment['machines2fail']
                
            row_start = comp_assignment['row_sets'][i]
            row_stop  = comp_assignment['row_sets'][i+1]
            for j in range(L):
                for machine in machine_set:
                    z[row_start+q_coded*j:row_stop+q_coded*j] += H_code[machine_set][machine][j]*comp_coded[machine][i]
    
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[4] += timeStamps[timeCount]-timeStamps[timeCount-1] 
    
    ####################################
    # Adjust Worker Assignments and Send
    ####################################
        
    if rank == 0:
        if option != 'Uncoded':
            comp_assignment['row_sets'] = [int(i) for i in np.array(comp_assignment['row_sets'])*float(r_coded)/float(q_coded)]
        else:
            comp_assignment['row_sets'] = [0, r_coded]
            
        comp_assignment['w'] = np.copy(z)
        
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[5] += timeStamps[timeCount]-timeStamps[timeCount-1] 
    
    comp_assignment = comm.bcast(comp_assignment,root=0)
    
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[6] += timeStamps[timeCount]-timeStamps[timeCount-1] 
        # The error is the loss function
        Err[t] = 0.5*np.linalg.norm(z)**2 + 0.5*gamma*np.linalg.norm(x)**2
        if t % 10 == 0:
            print Err[t]
            run_file.write('%.2f \n' %Err[t])
    
    #############################################
    # Workers Perform Assigned Coded Computations
    #############################################
    
    comp_coded = {}
    
    t3 = time.time()
    
    if rank not in comp_assignment['machines2fail']:
        for i in range(len(comp_assignment['machines'])):
            dec_set = comp_assignment['machines'][i]
            if rank in dec_set and rank > 0:
                comp_coded[i] = np.matmul(
                                    A_coded_T[comp_assignment['row_sets'][i]:comp_assignment['row_sets'][i+1]],
                                    comp_assignment['w'])
    
            
    comp_coded['inv_time'] = 1/(t2 + time.time() - t3)
    
    comm.Barrier()
    
    if rank == 0:
        timeCount += 1
        timeStamps[timeCount] = time.time()
        timeTrack[7] += timeStamps[timeCount]-timeStamps[timeCount-1] 
        
    ##################################################
    # Master Gathers the Computations from the Workers
    ##################################################
    
    comp_coded = comm.gather(comp_coded,root=0)
    
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        timeTrack[8] += timeStamps[timeCount]-timeStamps[timeCount-1]      
        
    #################################################
    # Master Combines Worker Results for Final Result
    #################################################
    
    if rank == 0:
        # Master node
        x -= eta*gamma*x
        for i in range(len(comp_assignment['machines'])):
            machine_set = frozenset(np.random.choice(
            list(comp_assignment['machines'][i] - comp_assignment['machines2fail']),
            L,replace=False))
            row_start = comp_assignment['row_sets'][i]
            row_stop  = comp_assignment['row_sets'][i+1]
            for j in range(L):
                for machine in machine_set:
                    x[row_start+r_coded*j:row_stop+r_coded*j] -= eta * H_code[machine_set][machine][j]*comp_coded[machine][i]
    
    if rank == 0:
        timeCount +=1
        timeStamps[timeCount] = time.time()
        iteration_time[t] = timeStamps[timeCount] - timeStamps[2]
        timeTrack[9] += timeStamps[timeCount]-timeStamps[timeCount-1] 

    #######################
    # Master Updates Speeds
    #######################     
    
    if rank == 0 and option != 'Uncoded' and t<T-1:
        mean_stable_speed = float(0)
        stable_machines = set([n for n in range(1,K+1)]) - comp_assignment['machines2fail']
        num_stable_machines = len(stable_machines)
        for machine in stable_machines:
            mean_stable_speed += comp_load[machine-1]*comp_coded[machine]['inv_time']/num_stable_machines
        for machine in set(available_machines)-comp_assignment['machines2fail']:
            speed_comp[machine-1] = speed_update_factor*comp_load[machine-1]*comp_coded[machine]['inv_time']/mean_stable_speed \
                                    + (1-speed_update_factor)*speed_comp[machine-1]      

        
if rank == 0:

    print np.round(100*speed_comp)/100
    run_file.write(np.array2string(speed_comp,formatter={'float_kind':lambda x: "%.2f" % x})+' \n')
    
    np.save('times.npy',iteration_time)
    np.save('error.npy',Err)    

    print '\nMaster:'
    print '\t Mean Comp Assign: \t %.2f ms' % (1000*timeTrack[0]/float(T)) 
    run_file.write('\t Mean Comp Assign: \t %.2f ms \n' % (1000*timeTrack[0]/float(T)) )    
    
    print '\nMaster:'
    print '\t Mean Comp Send: \t %.2f ms' % (1000*timeTrack[1]/float(T))   
    run_file.write('\t Mean Comp Send: \t %.2f ms \n' % (1000*timeTrack[1]/float(T))   ) 
    
    print '\nMaster:'
    print '\t Mean Worker Comp: \t %.2f ms' % (1000*timeTrack[2]/float(T))    
    run_file.write('\t Mean Worker Comp: \t %.2f ms \n' % (1000*timeTrack[2]/float(T))   )  
    
    print '\nMaster:'
    print '\t Mean Receive: \t  \t %.2f ms' % (1000*timeTrack[3]/float(T)) 
    run_file.write('\t Mean Receive: \t  \t %.2f ms \n' % (1000*timeTrack[3]/float(T)) )    
    
    print '\nMaster:'
    print '\t Mean Combine Results: \t %.2f ms' %( 1000*timeTrack[4]/float(T))
    run_file.write('\t Mean Combine Results: \t %.2f ms \n' %( 1000*timeTrack[4]/float(T)))
    
    print '\nMaster:'
    print '\t Mean Comp Adjust: \t %.2f ms' % (1000*timeTrack[5]/float(T))     
    run_file.write('\t Mean Comp Adjust: \t %.2f ms \n' % (1000*timeTrack[5]/float(T)) )
    
    print '\nMaster:'
    print '\t Mean Comp Send: \t %.2f ms' % (1000*timeTrack[6]/float(T))  
    run_file.write('\t Mean Comp Send: \t %.2f ms \n' % (1000*timeTrack[6]/float(T))  )  
    
    print '\nMaster:'
    print '\t Mean Worker Comp: \t %.2f ms' % (1000*timeTrack[7]/float(T))    
    run_file.write('\t Mean Worker Comp: \t %.2f ms \n' % (1000*timeTrack[7]/float(T)))  
    
    print '\nMaster:'
    print '\t Mean Receive: \t  \t %.2f ms' % (1000*timeTrack[8]/float(T))    
    run_file.write('\t Mean Receive: \t  \t %.2f ms \n' % (1000*timeTrack[8]/float(T)) ) 
    
    print '\nMaster:'
    print '\t Mean Combine Results: \t %.2f ms' %( 1000*timeTrack[9]/float(T)) 
    run_file.write('\t Mean Combine Results: \t %.2f ms \n' %( 1000*timeTrack[9]/float(T)) ) 

    print '\nMaster:'
    print '\t Mean Total Time: \t %.2f ms' %( 1000*float(timeStamps[timeCount]-timeStamps[2] )/float(T))
    run_file.write('\t Mean Total Time: \t %.2f ms \n' %( 1000*float(timeStamps[timeCount]-timeStamps[2] )/float(T)))

    print '\nMaster:'
    print '\t Computation Rate: \t %.2f' %( float(T) / float(timeStamps[timeCount]-timeStamps[2] )) 
    run_file.write('\t Computation Rate: \t %.2f \n' %( float(T) / float(timeStamps[timeCount]-timeStamps[2] )))
    
    run_file.close()