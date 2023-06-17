"""
Script to optimize CPFEM parameters using as the engine Abaqus and Huang's subroutine.
All user inputs should be in the ``opt_input.py`` file.
Two import modes: tries to load Abaqus modules (when running to extract stress-strain
info) or else imports sciki-optimize library (when running as main outside of Abaqus).
"""
import os
import shutil
import numpy as np
import opt_input as uset  # user settings file in run folder

from matmdl.experimental import ExpData
from matmdl.optimizer import InOpt
from matmdl.optimizer import instantiate_optimizer, get_next_param_set, write_opt_progress, update_progress, load_opt
from matmdl.runner import get_first, remove_out_files, combine_SS, write_params, refine_run
from matmdl.crystalPlasticity import get_orient_info, load_subroutine, param_check
from matmdl.engines import job_run, job_extract, check_complete
from matmdl.objectives import write_error_to_file, write_maxRMSE, calc_error, max_rmse


def main():
    """Instantiate data structures, start optimization loop."""
    remove_out_files()
    global exp_data, in_opt
    # TODO declare out_progress global up here?
    exp_data = ExpData(uset.orientations)
    in_opt = InOpt(uset.orientations, uset.params)
    opt = instantiate_optimizer(in_opt, uset)
    if uset.do_load_previous: opt = load_opt(opt)
    load_subroutine()

    loop(opt, uset.loop_len)


def loop(opt, loop_len):
    """Holds all optimization iteration instructions."""
    def single_loop(opt, i):
        """
        Run single iteration (one parameter set) of the optimization scheme.

        Single loops need to be separate function calls to allow empty returns to exit one
        parameter set.
        """
        global opt_progress  # global progress tracker, row:(i, params, error)
        next_params = get_next_param_set(opt, in_opt)
        write_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
        while param_check(uset.params):  # True if Tau0 >= TauS
            # this tells opt that params are bad but does not record it elsewhere
            opt.tell(next_params, max_rmse(i))
            next_params = get_next_param_set(opt, in_opt)
            write_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
        else:
            for orient in uset.orientations.keys():
                if in_opt.has_orient_opt[orient]:
                    orient_components = get_orient_info(next_params, orient, in_opt)
                    write_params('mat_orient.inp', orient_components['names'], orient_components['values'])
                else:
                    shutil.copy(uset.orientations[orient]['inp'], 'mat_orient.inp')
                shutil.copy('{0}_{1}.inp'.format(uset.jobname, orient), '{0}.inp'.format(uset.jobname))
                
                job_run()
                if not check_complete(): # try decreasing max increment size
                    refine_run()
                if not check_complete(): # if it still fails, write max_rmse, go to next parameterset
                    write_maxRMSE(i, next_params, opt, in_opt)
                    return
                else:
                    output_fname = 'temp_time_disp_force_{0}.csv'.format(orient)
                    if os.path.isfile(output_fname): 
                        os.remove(output_fname)
                    job_extract(orient)  # extract data to temp_time_disp_force.csv
                    if np.sum(np.loadtxt(output_fname, delimiter=',', skiprows=1)[:,1:2]) == 0:
                        write_maxRMSE(i, next_params, opt, in_opt)
                        return

            # error value:
            rmse_list = []
            for orient in uset.orientations.keys():
                rmse_list.append(calc_error(exp_data.data[orient]['raw'], orient))
                combine_SS(zeros=False, orientation=orient)  # save stress-strain data
            write_error_to_file(rmse_list, in_opt.orients)
            rmse = np.mean(rmse_list)
            opt.tell(next_params, rmse)
            opt_progress = update_progress(i, next_params, rmse)
            write_opt_progress(in_opt)
    
    get_first(opt, in_opt)
    for i in range(loop_len):
        single_loop(opt, i)


if __name__ == '__main__':
    main()
