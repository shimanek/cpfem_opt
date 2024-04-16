"""
Choose the computational engine for running, checking, and extracting
finite element job information. Currently only Abaqus exists.
"""

from matmdl.core.parser import uset

match uset.format:
	case "huang":
		from .abaqus import (
			extract,
			file_patterns,
			has_completed,
			pre_run,
			prepare,
			run,
			write_strain,
		)
	case "fepx":
		from .fepx import (
			extract,
			file_patterns,
			has_completed,
			pre_run,
			prepare,
			run,
			write_strain,
		)
