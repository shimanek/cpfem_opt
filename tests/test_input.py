import unittest
from matmdl.parser import uset
import numpy as np
import os


class TestExp(unittest.TestCase):
	def test_data_limits(self):
		from matmdl.experimental import ExpData
		exp = ExpData(uset.orientations)
		data_out = np.loadtxt("exp_out.csv", delimiter=",")
		parsed_data = exp.data['test']['raw']
		equal_elements = data_out==parsed_data
		self.assertTrue(equal_elements.all())
		os.remove("temp_expSS.csv")


class TestError(unittest.TestCase):
	def diff_test_linear(self, b1, m1, b2, m2, err_stress=None, err_slope=None):
		from matmdl.objectives.rmse import _stress_diff, _slope_diff
		x = np.linspace(0.01, 1.0, 100)
		def curve1(x):
			return b1 + m1*x
		def curve2(x):
			return b2 + m2*x
		diff_stress = _stress_diff(x, curve1, curve2)
		diff_slope = _slope_diff(x, curve1, curve2)
		if err_stress is not None:
			try:
				self.assertTrue(np.abs(diff_stress - err_stress) < 1e-6)
			except AssertionError as e:
				print("\nError in linear diff test stress with:")
				print("linear info:", b1, m1, b2, m2)
				print(f"expected err_stress of {err_stress}, got {diff_stress}")
				raise e
			self.assertTrue(np.abs(diff_slope - err_slope) < 1e-6)
		if err_slope is not None:
			try:
				self.assertTrue(np.abs(diff_slope - err_slope) < 1e-6)
			except AssertionError as e:
				print("\nError in linear diff test stress with:")
				print("linear info:", b1, m1, b2, m2)
				print(f"expected err_stress of {err_slope}, got {diff_slope}")
				raise e

	def test_diff(self):
		self.diff_test_linear(b1=0, m1=5, b2=0, m2=10, err_stress=50, err_slope=50)
		self.diff_test_linear(b1=0, m1=5, b2=2, m2=5, err_slope=0)	


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
