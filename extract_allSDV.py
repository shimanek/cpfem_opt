number_of_elements = 7
### ^ important user input
'''
Script to extract displacement (U2), either reaction force (RF2) 
or concentrated force (CF2), SDV1-24, and SDV121.
Written for a geometric model consisting of a line of elements, 
extracting data from the central one.
Call with abaqus cae noGUI=extract.py
Date: 2020-05-14
Author: J. Shimanek
'''
from abaqusConstants import *
from abaqus import *
import visualization
import numpy as np
import xyPlot
import os

# variable names:
nodal_variable = ['U', 'RF']  # can change RF to CF if needed
nodal_component = [ var + '2' for var in nodal_variable ]
sdv_nums = list(range(1,121))  # full STATEV array
# sdv_nums.insert(0,121) # 121,1,2,3,...,24
sdv_list = ['SDV' + str(num) for num in sdv_nums]
midpoint = int((number_of_elements+1)/2)

# delete old files: 
for f in sdv_list + nodal_component:
    file_name = 'dat_out_' + f + '.txt'
    if os.path.isfile( file_name ):
        os.remove( file_name )

# get odb name (will take last if multiple)
for file in os.listdir(os.getcwd()):
        if file.endswith('.odb'):
            dbName = os.path.join(os.getcwd(), file)

# load odb and extract xy data to variables 
o1 = session.openOdb(name=dbName)
session.viewports['Viewport: 1'].setValues(displayedObject=o1)

# get nodal values: U2, (either RF2 or CF2)
nodal_arr_list = [None] * len(nodal_variable)
for k in range(2):
    temp_list = [None] * 8
    for i in range(8):
        temp_list[i] = session.xyDataListFromField(odb=o1, outputPosition=NODAL, 
            variable=((nodal_variable[k], NODAL, ((COMPONENT, nodal_component[k]), )), ), 
            nodeLabels=(('PART-1-1', (str(i+1), )), ))[0]
    # time, length of arrays same across all:
    time = np.asarray(temp_list[0])[:,0]
    length = np.shape(time)[0]
    # get all values in an array: 
    temp_arr = np.zeros( [length,8] )
    for i in range(8):
        for j in range(length):
            temp_arr[j,i] = temp_list[i][j][1]
    nodal_arr_list[k] = temp_arr
    # np.savetxt('dat_out_' + nodal_component[k] + '.txt', temp_arr)
# np.savetxt('dat_out_time.txt', time)

# extract SDVs from odb:
for i in sdv_nums:
    session.xyDataListFromField(odb=o1, outputPosition=INTEGRATION_POINT, 
        variable=(('SDV' + str(i), INTEGRATION_POINT),),
        elementLabels=(('PART-1-1', (str(midpoint), )), ))

# combine files into a numpy binary array:
# 8 cols each: sdv1-121, rf2, u2, fraction of loading step = length x 8 x ( 121 + 3 = 124 )
dat_out_combined = np.zeros( [length,8,124] )
index_count = 0
for sdv in sdv_list:
    temp_list = [None] * 8
    for i in range(8):
        temp_list[i] = session.xyDataObjects[sdv + ' PI: PART-1-1 E: ' + str(midpoint) + ' IP: ' + str(i+1)]
    temp_arr = np.zeros( [length,8] )
    for i in range(8):
        for j in range( length ):
            temp_arr[j,i] = temp_list[i][j][1]
    # np.savetxt( 'dat_out_' + sdv + '.txt', temp_arr ) 
    dat_out_combined[:,:,index_count] = temp_arr
    index_count += 1
dat_out_combined[:,:,-1] = np.hstack( ( [ np.reshape( time, (length,1) ) ] * 8 ) )  # time, all columns same
dat_out_combined[:,:,-2] = nodal_arr_list[0]  # U2
dat_out_combined[:,:,-3] = nodal_arr_list[1]  # RF2 or CF2
np.save('out_combined.npy', dat_out_combined)
out_nodeAverage = np.mean( dat_out_combined, axis=1 )
np.save('out_average.npy', out_nodeAverage )
o1.close()

