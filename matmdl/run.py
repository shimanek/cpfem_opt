"""
Runnable module to start an optimization run. 
All input should be in an `input.toml` file in the directory where this is called.
"""
import os
import shutil
import numpy as np

from matmdl.experimental import ExpData
from matmdl.optimizer import InOpt
from matmdl.optimizer import instantiate_optimizer, get_next_param_set, load_opt, update_optimizer_if_needed
from matmdl.runner import get_first, remove_out_files, write_input_params, refine_run, check_single
from matmdl.crystalPlasticity import get_orient_info, load_subroutine
from matmdl.engines import job_run, job_extract, check_complete
from matmdl.objectives import calc_error
from matmdl.writer import write_error_to_file, combine_SS, write_params_to_file
from matmdl.parser import uset
from matmdl.parallel import check_parallel, Checkout, update_parallel
from matmdl.state import state


def main():
    """
    Instantiate data structures, start optimization loop. 

    Checks for single run option, which runs then exits. 
    Checks if current process is part of a parallel pool. 
    Checks if previous output should be reloaded. 
    """
    check_single()
    check_parallel()
    remove_out_files()
    global exp_data, in_opt
    exp_data = ExpData(uset.orientations)
    in_opt = InOpt(uset.orientations, uset.params)
    opt = instantiate_optimizer(in_opt, uset)
    if uset.do_load_previous: opt = load_opt(opt)
    load_subroutine()

    loop(opt, uset.loop_len)


def loop(opt, loop_len):
    """Holds all optimization iteration instructions."""
    def single_loop(opt):
        """
        Run single iteration (one parameter set) of the optimization scheme.

        Single loops need to be separate function calls to allow empty returns to exit one
        parameter set.
        """
        next_params = get_next_param_set(opt, in_opt)
        write_input_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])

        with state.TimeRun()():
            for orient in in_opt.orients:
                # TODO: below block group and replace
                if in_opt.has_orient_opt[orient]:
                    orient_components = get_orient_info(next_params, orient, in_opt)
                    write_input_params('mat_orient.inp', orient_components['names'], orient_components['values'])
                else:
                    if 'inp' in uset.orientations[orient]:
                        shutil.copy(uset.orientations[orient]['inp'], 'mat_orient.inp')
                try:
                    shutil.copy('{0}_{1}.inp'.format(uset.jobname, orient), '{0}.inp'.format(uset.jobname))
                except: # todo: get exception or redo this whole logic block
                    pass

                job_run()
                if not check_complete(): # try decreasing max increment size
                    refine_run()
                if not check_complete(): # if it still fails, tell optimizer a large error, continue
                    opt.tell(next_params, uset.large_error)
                    print(f"Warning: early incomplete run for {orient}, skipping to next paramter set")
                    return
                else:
                    output_fname = 'temp_time_disp_force_{0}.csv'.format(orient)
                    if os.path.isfile(output_fname): 
                        os.remove(output_fname)
                    job_extract(orient)  # extract data to temp_time_disp_force.csv
                    if np.sum(np.loadtxt(output_fname, delimiter=',', skiprows=1)[:,1:2]) == 0:
                        opt.tell(next_params, uset.large_error)
                        print(f"Warning: early incomplete run for {orient}, skipping to next paramter set")
                        return

        # write out:
        update_params, update_errors = [], []
        with Checkout("out"):
            # check parallel instances:
            update_params_par, update_errors_par = update_parallel()
            if len(update_errors_par) > 0:
                update_params = update_params + update_params_par
                update_errors = update_errors + update_errors_par

            # this instance:
            errors = []
            for orient in in_opt.orients:
                errors.append(calc_error(exp_data.data[orient]['raw'], orient))
                combine_SS(zeros=False, orientation=orient)  # save stress-strain data

            mean_error = np.mean(errors)  #TODO can be handled within error
            update_params = update_params + [next_params]
            update_errors = update_errors + [mean_error]

            # write this instance to file:
            write_error_to_file(errors, in_opt.orients)
            write_params_to_file(next_params, in_opt.params)

        # update optimizer outside of Checkout context to lower time using output files:
        update_optimizer_if_needed(opt, update_params, update_errors)


    get_first(opt, in_opt)
    for _ in range(loop_len):
        single_loop(opt)


if __name__ == '__main__':
    main()
