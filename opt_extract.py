"""
ODB Extraction - Shipin Qin, 2018-06
"""

from odbAccess import *
from abaqusConstants import *

from odbMaterial import *
from odbSection import *

class Get_Fd(object):
    
    def __init__(self,ResultFile):

        CurrentPath = os.getcwd()
        self.ResultFilePath = os.path.join(CurrentPath, ResultFile + '.odb')

        self.Time = []
        self.TopU2 = []
        self.TopRF2 = []
        
        step = 'Loading'
        instance = 'PART-1-1'
        TopRPset = 'RP-TOP'
        
        odb = openOdb(path=self.ResultFilePath, readOnly=True)
        steps = odb.steps[step]
        frames = odb.steps[step].frames
        numFrames = len(frames)
        TopRP = odb.rootAssembly.instances[instance].nodeSets[TopRPset] # if the node set is in Part
        #TopNodes = odb.rootAssembly.nodeSets[NodeSetTop] # if the node set is in Assembly
        
        for x in range(numFrames):
            Frame = frames[x]
            # Record time
            Time1 = Frame.frameValue
            self.Time.append(Time1) # list append
            # Top RP results
            Displacement = Frame.fieldOutputs['U']
            ReactionForce = Frame.fieldOutputs['RF']
            TopU  = Displacement.getSubset(region=TopRP).values
            TopRf = ReactionForce.getSubset(region=TopRP).values
            self.TopU2  = self.TopU2  + map(lambda x:x.data[1], TopU)  # list combination
            self.TopRF2 = self.TopRF2 + map(lambda x:x.data[1], TopRf) # list combination
        
        odb.close()

def write2file():
    job = [f for f in os.listdir(os.getcwd()) if f.endswith('.odb')][0][:-4]
    Result_Fd = Get_Fd(job)
    with open('temp_time_disp_force.csv','w') as f:
        f.write('{0},{1},{2}\n'.format('Time','U2','RF2'))
        for i in range(len(Result_Fd.Time)):
            f.write('%.5f,' % Result_Fd.Time[i])
            f.write('%.5f,' % Result_Fd.TopU2[i])
            f.write('%.5f\n' % Result_Fd.TopRF2[i])