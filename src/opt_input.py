"""
User settings for opt_fea.py and opt_plot.py
"""
import os
param_list = ['Tau01', 'H01', 'TauS1', 'Tau02', 'H02', 'TauS2']
param_bounds = [(100,200), (1400,2400), (400,800), (500,800), (4100,6000), (100,300)]
loop_len = 15
n_initial_points = 5
large_error = 5e3
# ^ backup RMSE of runs which don't finish; first option uses 1.5 * IQR(first few RMSE)
exp_SS_file = [f for f in os.listdir(os.getcwd()) if f.startswith('exp')][0]
param_file = [f for f in os.listdir(os.getcwd()) if f.startswith('mat_param')][0]
length = 9
area = 9 * 9
jobname = 'UT_mX'
recursion_depth = 1
max_strain = 0.0
# ^ 0 for max exp value, fractional strain (0.01=1%) otherwise
i_powerlaw = 0  # interpolation: 0 = linear, 1 = Holomon equation
umat = 'umatcrystal_mod_XIT.f'
cpus = 4
do_load_previous = False

# plot settings:
grain_size_name = 'mX'  # string
title = '9x9x9el-3x3x3gr model'
param_additional_legend = ['q']