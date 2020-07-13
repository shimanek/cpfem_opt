'''
test script to optimize a single CPFEM parameter
Date: June 30, 2020
'''

import time
import os
import subprocess
import sys
import numpy as np
from skopt import Optimizer
from scipy.interpolate import interp1d
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt


opt = Optimizer(
    dimensions=[ (1, 30) ], 
    base_estimator='gp' 
)

opt_progress = np.array([0.0,0.0])

for j in range(2):

    next_tau0 = opt.ask()
    opt_progress = np.vstack( (opt_progress, np.array([next_tau0[0],0])) )

    # place new value in Mat_*.inp file
    # for test purposes, just put at end of file
    f1 = open( 'Mat_BW_27grains.inp', 'r' )
    f2 = open( 'temp_mat_file.inp', 'w+' )
    lines = f1.readlines()
    f2.writelines( lines[ :len(lines)-1 ])
    f2.write( 'Tau0 = ' + str( next_tau0[0] ) )
    f1.close()
    f2.close()

    os.remove( 'Mat_BW_150grains.inp' )
    os.rename( 'temp_mat_file.inp', 'Mat_BW_150grains.inp' )

    # submit job 
    os.system( 'abaqus job=UT_150grains user=umatcrystal_mod_XIT.f cpus=4 double int ask_delete=OFF' )

    # get RMSE
    time.sleep( 5 )
    os.system( 'abaqus cae nogui=extract_SS_singleEl.py' )
    time.sleep( 5 )
    simSS = np.loadtxt( 'allArray.csv', delimiter=',', skiprows=1 )[:,1:]
    simSS[:,0]/3  # disp to strain
    simSS[:,1]/(3**2)  # force to stress

    # smooth out simulated SS
    smoothenedSS = interp1d( simSS[:,0], simSS[:,1] )

    # load experimental data
    expSS = np.loadtxt( 'expSS130um.txt', skiprows=2 )

    # error function
    deviations = np.asarray( [ smoothenedSS(expSS[i,0]) - expSS[i,1] for i in range(len(expSS)) ] )
    rmse = np.sqrt( np.sum( deviations**2 ) / len(expSS) ) 

    opt_progress[-1,1] = rmse

    res = opt.tell( next_tau0, rmse )

    np.savetxt('out_progress.txt',opt_progress, delimiter='\t')

conv = plot_convergence(res)
plt.savefig('convergence.png',dpi=300)
plt.close()
