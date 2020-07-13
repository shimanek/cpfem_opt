'''
About: Python runner script for CPFEM Abaqus optimization jobs
Date: 2020-06-24
'''
import time
import os
import subprocess
import sys
import numpy as np
# np.random.seed(1111)
from skopt import Optimizer


start = time.time()
with open('qrunner') as f:
    content = f.readlines()
content = [ x.strip() for x in content ]
for line in content:
    wall_ind = line.find( 'walltime' )
    if wall_ind > -1:
        wall = line[17:]
walltime = float(wall[0:2]) + float(wall[3:5])/60 + float(wall[6:8])/3600

job_key = 'UT'
os.system( 'qsub qabqs_opt1' )

while ( time.time() - start )/3600 < walltime * 0.8:
    last_line = ''
    stafile = [ f for f in os.listdir() if f.startswith( job_key ) ][0]
    stafile = stafile[ 0 : stafile.find('.') ] + '.sta'

    if os.path.isfile( stafile ): os.remove( stafile )

    # while last_line != ' THE ANALYSIS HAS COMPLETED SUCCESSFULLY':
    while not ( 'SUCCESSFULLY' in last_line ):

        # if runner time is too low, resubmit runner, kill this instance
        if ( time.time() - start )/3600 > walltime - 5/60:
            os.system( 'qsub qrunner' )
            sys.exit() 
            # TODO when restarting, runner will delete, resub current sim 

        # wait and update last line
        time.sleep( 30 )
        print( 'checked tail' )
        if os.path.isfile( stafile ):
            # last_line = os.system( 'tail -n 1 ' + stafile )
            last_line = str( subprocess.check_output( ['tail', '-1', stafile] ) )

    # parse results
    time.sleep( 5 )
    os.system( 'abaqus cae nogui=extract_allSDV.py' )
    time.sleep( 5 )
    os.system( 'abaqus cae nogui=extract_SS_singleEl.py' )
    time.sleep( 5 )

    # optimization section
    # opt = Optimizer( [0, eng_strain], base_estimator='GP', acq_optimizer='sampling')
    # next_tau0 = opt.ask()
    # get error 


    # change input parameters

    # qsub new job
    os.system( 'qsub qabqs_opt1' )

    # qsub new runner, kill this runner
os.system( 'qsub qrunner' )
