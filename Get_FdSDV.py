# This code extracts the reaction force and displacements on reference nodes in RVE simulations.
# Shipin Qin, 06. 05, 2018

from odbAccess import *
from abaqusConstants import *

from odbMaterial import *
from odbSection import *

import csv
import math

import os


class Get_Fd(object):
    
    def __init__(self,ResultFile, y0=1.,x0=1.,z0=0.005, E=201000., nu=0.3):
        
        print("\n***Get reaction force and displacement from reference point \"RP-TOP\"\n")
        CurrentPath = os.getcwd()
        self.ResultFilePath = CurrentPath+'/'+ResultFile+'.odb'

        self.Time = []
        self.TopU2 = []
        self.TopRF2 = []
        self.TopU1 = []
        
        step = 'Loading'
        instance = 'PART-1-1'
        TopRPset = 'RP-TOP'
        RightRPset = 'RP-RIGHT'
        CenterNodeset = 'CENTERNODE'
        CenterNodeset_Surf = 137      # The surface center node number in half thickness butterfly simulation.
        
        # Set up RPs
        odb = openOdb(path=self.ResultFilePath, readOnly=True) #if the odb file is in your working directory
        steps = odb.steps[step]
        frames = odb.steps[step].frames
        numFrames = len(frames)
        TopRP = odb.rootAssembly.instances[instance].nodeSets[TopRPset] # if the node set is in Part
        #TopNodes = odb.rootAssembly.nodeSets[NodeSetTop] # if the node set is in Assembly
        
        for x in range(numFrames):
            # Display extraction progress
            progress = (float(x)*100)//float(numFrames)
            print("--%d %% \r" % progress),
            Frame = frames[x]
            # Record time
            Time1 = Frame.frameValue
            self.Time.append(Time1) # list append
            # Top RP results
            Displacement = Frame.fieldOutputs['U']
            ReactionForce = Frame.fieldOutputs['RF']
            TopU  = Displacement.getSubset(region=TopRP).values
            TopRf = ReactionForce.getSubset(region=TopRP).values
            self.TopU2       = self.TopU2  + map(lambda x:x.data[1], TopU) # list combination
            self.TopU1       = self.TopU1  + map(lambda x:x.data[0], TopU) # list combination
            self.TopRF2      = self.TopRF2 + map(lambda x:x.data[1], TopRf) # list combination
        
        odb.close()

class Get_SDVs(object):
    
    def __init__(self,ResultFile, y0=1.,x0=1.,z0=0.005, E=201000., nu=0.3):
        
        print("\n***Get SDVs from integration point 1 in element set \"All\"\n")
        CurrentPath = os.getcwd()
        self.ResultFilePath = CurrentPath+'/'+ResultFile+'.odb'

        self.Time = []
        self.SDV_values = []
        
        step = 'Loading'
        instance = 'PART-1-1'
        ElSet    = 'ALL'
        # SDV_lables = ['SDV' + str(i+1) for i in range(24)] + ['SDV121']
        # SDV_lables = ['SDV121']
        SDV_lables = ['SDV' + str(i+1) for i in range(12)] + ['SDV' + str(i+1+12*9) for i in range(12)] + ['SDV121']
        
        # Set up element subregions
        odb        = openOdb(path=self.ResultFilePath, readOnly=True) #if the odb file is in your working directory
        steps      = odb.steps[step]
        frames     = odb.steps[step].frames
        numFrames  = len(frames)
        El         = odb.rootAssembly.instances[instance].elementSets[ElSet]
        
        for x in range(numFrames):
            # Display extraction progress
            progress = (float(x)*100)//float(numFrames)
            print("--%d %% \r" % progress),
            Frame = frames[x]
            # Record time
            Time1 = Frame.frameValue
            self.Time.append(Time1) # list append
            # Element integration point results
            SDVs = [Frame.fieldOutputs[x].getSubset(region=El, position=INTEGRATION_POINT).values[0] for x in SDV_lables]   # values[0] indicates integration point 1
            
            self.SDV_values  = self.SDV_values + [[x.data for x in SDVs]]
        
        odb.close()

#~ FixedPath = '/gpfs/scratch/svq5030/RVE_Idealized_3DPS_MMC-Ostlund_MaxS-Uthai-Rama_CZM-Ramazani_CohE0p16/'
#FixedPath = os.getcwd()

# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
class Output(object):
    
    def write2scrn(self, Job):

        print('*** Input any key to print Fd results on the screen, input \"n\" or \"N\" to cancel')
        x = raw_input()
        if x != ('n' or 'N'):
            Result_Fd = Get_Fd(Job)
            for i in range(len(Result_Fd.Time)):
                print(Result_Fd.Time[i],',',Result_Fd.TopU2[i],',',Result_Fd.TopRF2[i])
            print(Job +' Get_Fd finished')

        print('*** Input any key to print SDV results on the screen, input \"n\" or \"N\" to cancel')
        x = raw_input()
        if x != ('n' or 'N'):
            Result_SDVs = Get_SDVs(Job)
            for i in range(len(Result_SDVs.Time)):
                print('%.4f,' % Result_SDVs.Time[i])
                for x in Result_SDVs.SDV_values[:][i]:
                    print('%.4f,' % x)
                print('')
            # print Result_SDVs.Time[i],',',Result_SDVs.TopU2[i],',',Result_SDVs.TopU1[i]
            print(Job +' Get_SDVs finished')
# --------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------
    def write2file(self, Job):

        print('*** Input any key to include SDV results, input \"n\" or \"N\" to cancel')
        xx = raw_input()
        f = open(Job+'_Result_temp.csv','w')
        print('\n'+Job+'results write to a file started')
        Result_Fd = Get_Fd(Job)
        if xx != ('n' or 'N'):
            Result_SDVs = Get_SDVs(Job)
        # Results = [Result_Fd.Time, Result_Fd.TopU2, Result_Fd.TopRF2]
        with open('A_Results_'+Job+'.csv','w') as f:
            f.write('{0},{1},{2},{3}\n'.format('Time','U2','RF2','SDV1-12, 109-120'))
            for i in range(len(Result_Fd.Time)):
                # f.write(Result_Fd.Time[i],',',Result_Fd.TopU2[i],',',Result_Fd.TopRF2[i]),
                f.write('%.4f,' % Result_Fd.Time[i]),
                f.write('%.4f,' % Result_Fd.TopU2[i]),
                f.write('%.4f,' % Result_Fd.TopRF2[i]),
                if xx != ('n' or 'N'):
                    for x in Result_SDVs.SDV_values[:][i]:
                        f.write('%.4f,' % x),
                f.write('\n')
            # f.write("{0},{1},{2},{3},{4},{5}\n".format('Time','U2','RF2','SDV1-12, 109-120'))
            # for x in zip(*Results):
            #     f.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(*x))
        print('\n'+Job+' results write to a file finished')

#Jobs = ['SingleCubicComp_Orient269_Ni_Aniso-S']
# select first .odb file in the folder:
Jobs = [f for f in os.listdir(os.getcwd()) if f.endswith('.odb')][0][:-4]
print('*** How do you want to output the result?')
print('-  1 - write to screen, 2 - write to a file')

x = input()
if x == 1:
    Output().write2scrn(Jobs)
elif x == 2:
    Output().write2file(Jobs)