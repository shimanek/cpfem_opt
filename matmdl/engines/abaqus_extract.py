"""
This runnable module requires Abaqus libraries and must be run
from Abaqus python. It extracts force-displacement data from
pre-defined reference points for use in comparison of model input
parameters to reference data.
"""

import os

from odbAccess import *
from abaqusConstants import *
from odbMaterial import *
from odbSection import *


def main():
	write2file()


class GetForceDisplacement(object):
	"""
	Open ODB file and store force-displacement data from a reference point.

	Args:
	    ResultFile (str): name of odb file without the '.odb'

	Attributes:
	    Time: simulation time (0->1)
	    TopU2: displacement of reference point
	    TopRF2: reaction force of reference point

	Note:
	    Depends on the Abaqus names of loading step, part instance, and reference point.
	    Requires Abaqus-specific libraries, must be called from Abaqus python.
	"""

	def __init__(self, ResultFile):
		CurrentPath = os.getcwd()
		self.ResultFilePath = os.path.join(CurrentPath, ResultFile + ".odb")

		self.Time = []
		self.TopU2 = []
		self.TopRF2 = []

		# Names that need to match the Abaqus simulation:
		step = "Loading"
		instance = "PART-1-1"
		TopRPset = "RP-TOP"

		odb = openOdb(path=self.ResultFilePath, readOnly=True)
		steps = odb.steps[step]
		frames = odb.steps[step].frames
		numFrames = len(frames)

		# if node set is in Part:
		TopRP = odb.rootAssembly.instances[instance].nodeSets[TopRPset]
		# if the node set is in Assembly:
		# TopNodes = odb.rootAssembly.nodeSets[NodeSetTop]

		for x in range(numFrames):
			Frame = frames[x]
			self.Time.append(Frame.frameValue)
			# Top RP results:
			Displacement = Frame.fieldOutputs["U"]
			ReactionForce = Frame.fieldOutputs["RF"]
			TopU = Displacement.getSubset(region=TopRP).values
			TopRf = ReactionForce.getSubset(region=TopRP).values
			# add to lists:
			self.TopU2 = self.TopU2 + map(lambda x: x.data[1], TopU)
			self.TopRF2 = self.TopRF2 + map(lambda x: x.data[1], TopRf)

		odb.close()


def write2file():
	"""
	Using GetForceDisplacement object, read odb file and write time-displacement-force to csv file.
	"""
	job = [f for f in os.listdir(os.getcwd()) if f.endswith(".odb")][0][:-4]
	Result_Fd = GetForceDisplacement(job)
	with open("temp_time_disp_force.csv", "w") as f:
		f.write("Time, U2, RF2\n")
		for i in range(len(Result_Fd.Time)):
			f.write("%.5f," % Result_Fd.Time[i])
			f.write("%.5f," % Result_Fd.TopU2[i])
			f.write("%.5f\n" % Result_Fd.TopRF2[i])


if __name__ == "__main__":
	main()
