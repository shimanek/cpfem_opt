"""
User settings for opt_fea.py and opt_plot.py
"""
import os

param_list = [
	'Tau01', 'H01', 'TauS1', 
	'Tau02', 'H02', 'TauS2'
]
param_bounds = [
	(100,200), (1400,2400), (400,800), 
	(500,800), (4100,6000), (100,300)
]
orientations = {
	'001':{
		'exp':'exp_W-mX-001.csv',
		'dir_load':(0,0,1),
		'offset':{
			'dir_0deg':(0,1,1),
			'mag_bounds':(0,1),
			'deg_bounds':(0,90)
		}
	},
	'111':{
		'exp':'exp_W-mX-111.csv',
		'dir_load':(1,1,1),
		'offset':{
			'dir_0deg':(0,1,1),
			'mag_bounds':(0,1),
			'deg_bounds':(0,90)
		}
	}
}
loop_len = 15
n_initial_points = 5
large_error = 1e2
# ^ backup RMSE of runs which don't finish; first option uses 1.5 * IQR(first few RMSE)
param_file = [f for f in os.listdir(os.getcwd()) if f.startswith('mat_param')][0]
length = 1
area = 1 * 1
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
title = ''
param_additional_legend = ['q1']