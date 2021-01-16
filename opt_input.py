### user input
param_list = ['Tau0', 'H0', 'TauS', 'hs', 'gamma0']
param_bounds = [ (1,100), (100,500), (1,200), (0,100), (0.0001,0.4) ]
loop_len = 150
n_initial_points = 50
large_error = 5e3  
# ^ backup RMSE of runs which don't finish; first option uses 1.5 * IQR(first few RMSE)
# exp_SS_file = [f for f in os.listdir(os.getcwd()) if f.startswith('exp')][0]
length = 9
area = 9 * 9
jobname = 'UT_729grains'
recursion_depth = 3
max_strain = 0.0
# ^ 0 for max exp value, fractional strain (0.01=1%) otherwise
### end input