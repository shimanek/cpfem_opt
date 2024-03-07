"""
Choose the computational engine for running, checking, and extracting 
finite element job information. Currently only Abaqus exists.
"""
from matmdl.parser import uset

match uset.format:
	case "huang":
		from .abaqus import run, extract, has_completed, prepare
	# case "fepx":
	# 	from .fepx import run, extract, has_completed, prepare
