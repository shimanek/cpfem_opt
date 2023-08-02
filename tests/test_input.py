import unittest
from matmdl.parser import uset
import numpy as np


class TestInput(unittest.TestCase):
	def test_input(self):
		self.assertTrue(uset.params['Tau0'] == [100,200])
		self.assertTrue(uset.orientations['001']['inp'] == 'mat_orient_100.inp')


class TestCP(unittest.TestCase):
	def test_rot(self):
		from matmdl.crystalPlasticity import get_offset_angle
		dir_in_load = np.array([1,2,3])
		dir_in_0deg = np.array([7,8,9])
		angle_in = 8.64489
		factor_out = get_offset_angle(dir_in_load, dir_in_0deg, angle_in)
		self.assertTrue(factor_out - 0.3 < 1e-6)


if __name__ == "__main__":
	unittest.main()
