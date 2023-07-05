import unittest
from matmdl.parser import uset


class TestGetAngles(unittest.TestCase):
	def test_input(self):
		self.assertTrue(uset.params['Tau0'] == [100,200])
		self.assertTrue(uset.orientations['001']['inp'] == 'mat_orient_100.inp')


if __name__ == "__main__":
	unittest.main()
