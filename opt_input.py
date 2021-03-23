"""
User settings for opt_fea.py and opt_plot.py
"""
import os
param_list = ['Tau0', 'H0', 'TauS', 'hs', 'gamma0']
param_bounds = [ (1,100), (100,500), (1,200), (0,100), (0.0001,0.4) ]
loop_len = 150
n_initial_points = 50
large_error = 5e3  
# ^ backup RMSE of runs which don't finish; first option uses 1.5 * IQR(first few RMSE)
exp_SS_file = [f for f in os.listdir(os.getcwd()) if f.startswith('exp')][0]
param_file = [f for f in os.listdir(os.getcwd()) if f.startswith('Mat_BW')][0]
length = 9
area = 9 * 9
jobname = 'UT_729grains'
recursion_depth = 2
max_strain = 0.0
# ^ 0 for max exp value, fractional strain (0.01=1%) otherwise
i_powerlaw = 0  # interpolation: 0 = linear, 1 = Holomon equation

# plot settings:
grain_size_name = '0.12'  # string
title = '9x9x9el-3x3x3gr model'
