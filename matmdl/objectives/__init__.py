"""
Choose the objective function form. Currently only rmse exists.
"""
from matmdl.objectives.rmse import calc_error, max_rmse
from matmdl.writer import write_error_to_file, write_maxRMSE
