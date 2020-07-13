# data extraction for single element model
# built off of recorded macro
# first load abaqus: module load abaqus
# then call with: abaqus cae noGUI=extract.py
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import os
import numpy as np
import csv

u3file = 'RPbottom-u3.txt'
rf3file = 'RPbottom-rf3.txt'


def Macro1():
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior

    # get odb name (will take last if multiple)
    for file in os.listdir(os.getcwd()):
            if file.endswith('.odb'):
                dbName = os.path.join(os.getcwd(), file)
    
    # load odb and extract xy data to variables 
    o1 = session.openOdb(name=dbName)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    u3 = session.xyDataListFromField(odb=o1, outputPosition=NODAL, variable=(('U', NODAL, ((COMPONENT, 'U2'), )), ), nodeSets=('PART-1-1.RP-TOP', ))
    rf3 = session.xyDataListFromField(odb=o1, outputPosition=NODAL, variable=(('RF', NODAL, ((COMPONENT, 'RF2'), )), ), nodeSets=('PART-1-1.RP-TOP', ))


    o1.close()
    session.writeXYReport(u3file,u3)
    session.writeXYReport(rf3file,rf3)



if os.path.isfile(u3file):
    os.remove(u3file)
if os.path.isfile(rf3file):
    os.remove(rf3file)


if os.path.isfile('dispForce.npy'):
    os.remove('dispForce.npy')

Macro1()

u3txt = np.loadtxt('RPbottom-u3.txt', skiprows=4)
rf3txt = np.loadtxt('RPbottom-rf3.txt', skiprows=4)


t = u3txt[:,0]
u3 =  u3txt[:,1]
rf3 = rf3txt[:,1]


allArr = np.transpose(np.vstack((t,u3,rf3)))


header = np.array(['t', 'u', 'rf'])

with open('allArray.csv', 'wb') as fh:
    writer = csv.writer(fh, delimiter=',')
    writer.writerow(header)
    np.savetxt(fh, allArr, delimiter=',')


for file in os.listdir(os.getcwd()):
            if file.endswith('.txt') & file.startswith('sdv'):
                os.remove(file)
os.remove('RPbottom-u3.txt')
os.remove('RPbottom-rf3.txt')

