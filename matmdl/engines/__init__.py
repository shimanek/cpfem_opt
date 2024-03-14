"""
Choose the computational engine for running, checking, and extracting 
finite element job information. Currently only Abaqus exists.
"""
from matmdl.core.parser import uset

match uset.format:
	case "huang":
		from .abaqus import run, extract, has_completed, prepare, write_strain, pre_run
	case "fepx":
		from .fepx import run, extract, has_completed, prepare, write_strain, pre_run
