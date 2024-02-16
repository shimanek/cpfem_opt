import unittest
from matmdl.parser import uset
import numpy as np
import os


class TestExp(unittest.TestCase):
	def _by_orientation_name(self, exp, orient_name):
		data_out = np.loadtxt(f"exp_{orient_name}.csv", delimiter=",")
		parsed_data = exp.data[orient_name]['raw']
		equal_elements = np.equal(data_out,parsed_data)
		self.assertTrue(equal_elements.all())

	def test_data_limits(self):
		from matmdl.experimental import ExpData
		exp = ExpData(uset.orientations)
		self._by_orientation_name(exp, "test")
		self._by_orientation_name(exp, "001")
		os.remove("temp_expSS.csv")


class TestError(unittest.TestCase):
	def diff_test_linear(self, b1, m1, b2, m2, err_stress=None, err_slope=None, tol_stress=None, tol_slope=None):
		# tolerances are RMSE in respective units
		from matmdl.objectives.rmse import _stress_diff, _slope_diff
		x = np.linspace(0.0, 1.0, 100)
		def curve1(x):
			return b1 + m1*x
		def curve2(x):
			return b2 + m2*x
		diff_stress = _stress_diff(x, curve1, curve2)
		diff_slope = _slope_diff(x, curve1, curve2)
		print("stress:", diff_stress)
		print("slope:", diff_slope)

		if err_stress is not None:
			try:
				self.assertTrue(np.abs(diff_stress - err_stress) < tol_stress)
			except AssertionError as e:
				print("\nError in linear diff test stress with:")
				print("linear info:", b1, m1, b2, m2)
				print(f"expected err_stress of {err_stress}, got {diff_stress}")
				raise e

		if err_slope is not None:
			try:
				self.assertTrue(np.abs(diff_slope - err_slope) < tol_slope)
			except AssertionError as e:
				print("\nError in linear diff test stress with:")
				print("linear info:", b1, m1, b2, m2)
				print(f"expected err_slope of {err_slope}, got {diff_slope}")
				raise e

	def test_diff(self):
		self.diff_test_linear(b1=0, m1=5, b2=0, m2=10, err_stress=58, err_slope=50, tol_stress=1, tol_slope=1)
		self.diff_test_linear(b1=0, m1=5, b2=2, m2=5, err_slope=0, tol_slope=1)


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
