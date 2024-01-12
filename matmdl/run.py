"""
Script to optimize CPFEM parameters using as the engine Abaqus and Huang's subroutine.
All user inputs should be in the ``opt_input.py`` file.
Two import modes: tries to load Abaqus modules (when running to extract stress-strain
info) or else imports sciki-optimize library (when running as main outside of Abaqus).
"""
import os
import shutil
import numpy as np

from matmdl.experimental import ExpData
from matmdl.optimizer import InOpt
from matmdl.optimizer import instantiate_optimizer, get_next_param_set, load_opt
from matmdl.runner import get_first, remove_out_files, write_input_params, refine_run, check_single
from matmdl.crystalPlasticity import get_orient_info, load_subroutine, param_check
from matmdl.engines import job_run, job_extract, check_complete
from matmdl.objectives import calc_error
from matmdl.writer import write_error_to_file, combine_SS, write_params_to_file
from matmdl.parser import uset
from matmdl.parallel import check_parallel, Checkout, update_parallel


def main():
    """Instantiate data structures, start optimization loop."""
    check_single()  # takes over, runs, exits
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
        while param_check(uset.params):  # True if Tau0 >= TauS
            # soft enforce non-hyperrectangular parameter space
            # this tells opt that params are bad but does not record it elsewhere
            opt.tell(next_params, uset.large_error)
            next_params = get_next_param_set(opt, in_opt)
            write_input_params(uset.param_file, in_opt.material_params, next_params[0:in_opt.num_params_material])
        else:
            for orient in uset.orientations.keys():
                # TODO: below block group and replace
                if in_opt.has_orient_opt[orient]:
                    orient_components = get_orient_info(next_params, orient, in_opt)
                    write_input_params('mat_orient.inp', orient_components['names'], orient_components['values'])
                else:
                    shutil.copy(uset.orientations[orient]['inp'], 'mat_orient.inp')
                shutil.copy('{0}_{1}.inp'.format(uset.jobname, orient), '{0}.inp'.format(uset.jobname))

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
            with Checkout("out"):
                errors = []
                for orient in uset.orientations.keys():
                    errors.append(calc_error(exp_data.data[orient]['raw'], orient))
                    combine_SS(zeros=False, orientation=orient)  # save stress-strain data

                mean_error = np.mean(errors)  #TODO can be handled within error
                opt.tell(next_params, mean_error)

                update_parallel(opt)
                write_error_to_file(errors, in_opt.orients)
                write_params_to_file(next_params, in_opt.params)


    get_first(opt, in_opt)
    for _ in range(loop_len):
        single_loop(opt)


if __name__ == '__main__':
    main()
