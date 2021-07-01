"""
Make orthorhombic model with orthorhombic grains.
Inputs crystal plasticity parameters, assigns random orientations to all grains.
"""
from random import randrange
import numpy as np
import shutil
import sys
import os

#--------------------------------------------------------------------------------------------------
## preamble
#--------------------------------------------------------------------------------------------------
# data structures
class dim(object):
    """
    A place to store details of model dimensions.

    variables:
        edge_x, _y, _z  : edge length of model
        grain_x, _y, _z : edge length of grains within model
        eng_strain      : engineering strain to be applied to model (y dimension currently)
        disp            : displacement (calculated from strain) to be applied in y-direction
        num_nodes       : total number of nodes
        num_elements    : total number of elements in model
        num_grains      : total number of grains in model
        ref_node        : beginning number of nodes added as reference points
    """
class mesh(object):
    """
    For information about nodes and elements.

    variables:
        nodes       : array of node locations (x,y,z)
        elements    : array of elements (number of node 1, 2, ..., 8)
                      note that node numbers used in `elements` are 1-indexed
        right, left
        top, bottom
        front, back : node sets for the faces farthest in the directions +x, -x, +y, -y, +z, -z
    
    internal variables:
        nodes0,1,2    : column slices of `nodes` for faster searching
        rel_nodes     : array of 8 nearest node positions of current marker
        element_nodes : array of element numbers nearest to current marker
    """
class orient(object):
    """
    For orientation information.

    variables:
        max_index   : maximum Miller index for grain orientations

    internal variables:
        x      : [x,y,z] directions for local direction
        y      : [x,y,z] directions for 
    """
    def __init__(self):
        self.max_index = 7 
    def assign(self, x, y):
        self.x = x
        self.y = y
class files(object):
    """
    A place for filenames and the like.
    """
    def __init__(self):
        self.main       = 'job.inp'
        self.nodes      = 'Mesh_nodes.inp'
        self.nodeset    = 'Mesh_nset.inp'
        self.elements   = 'Mesh_elements.inp'
        self.elset      = 'Mesh_elset.inp'
        self.sections   = 'Mat_sects.inp'
        self.orients    = 'Mat_orient.inp'
        self.mesh_extra = 'Mesh_param.inp'
        self.material   = 'Mat_BW.inp'
        self.slip       = 'Mat_props.inp'
#--------------------------------------------------------------------------------------------------
# inputs
dim.edge_x = int(input('Enter edge length of cubic model (integer): '))
dim.edge_y = input('If cubic model, hit enter now. Else, enter second edge length (y): ')
if dim.edge_y == '':
    dim.edge_y = dim.edge_x
    dim.edge_z = dim.edge_x
else:
    dim.edge_y = int(dim.edge_y)
    dim.edge_z = int(input('Enter third edge length (z): '))

dim.grain_x = int(input( 'Enter the size of the grains in elements (x): ') )
dim.grain_y = input('If cubic model, hit enter now. Else, enter second grain dimension (y): ')
if dim.grain_y == '':
    dim.grain_y = dim.grain_x
    dim.grain_z = dim.grain_x
else:
    dim.grain_y = int(dim.grain_y)
    dim.grain_z = int(input('Enter third grain dimension (z): '))
dim.eng_strain = float(input('Input Engineering strain [default = 0.2]:  ') or '0.2')
dim.disp = dim.edge_y * dim.eng_strain
# instantiate filenames
files = files()
#--------------------------------------------------------------------------------------------------
# check if dimensions are appropriate
assert (dim.edge_x % dim.grain_x == 0), 'Dimension mismatch in x-direction'
assert (dim.edge_y % dim.grain_y == 0), 'Dimension mismatch in y-direction'
assert (dim.edge_z % dim.grain_z == 0), 'Dimension mismatch in z-direction'
#--------------------------------------------------------------------------------------------------
## Mesh
#--------------------------------------------------------------------------------------------------
# functions
def separate( row ):
    string_list = []
    for i in range(len(row)):
        string_list.append(str(row[i]))
    return string_list
#--------------------------------------------------------------------------------------------------
# nodes
dim.num_nodes = int( (dim.edge_x+1) * (dim.edge_y+1) * (dim.edge_z+1) ) 
mesh.nodes = np.empty( (dim.num_nodes, 3), dtype=float )
ct = 0
for z in range(0,dim.edge_z+1):
    for y in range(0,dim.edge_y+1):
        for x in range(0,dim.edge_x+1):
            mesh.nodes[ct, :] = [x, y, z]
            ct += 1
#--------------------------------------------------------------------------------------------------
# elements
mesh.nodes0 = np.asarray(mesh.nodes[:,0], dtype=int)
mesh.nodes1 = np.asarray(mesh.nodes[:,1], dtype=int)
mesh.nodes2 = np.asarray(mesh.nodes[:,2], dtype=int)
dim.num_elements = int( dim.edge_x * dim.edge_y * dim.edge_z )
mesh.elements = np.zeros( (dim.num_elements, 8), dtype=int ) 
ct_elements = 0
for z in range(0,dim.edge_z):
    for y in range(0,dim.edge_y):
        for x in range(0,dim.edge_x):
            marker = 0.5 * np.array([1,1,1]) + np.array([x,y,z])
            mesh.rel_nodes = marker + 0.5 * \
                np.array([  [-1,-1,-1],
                            [+1,-1,-1],
                            [+1,+1,-1],
                            [-1,+1,-1],
                            [-1,-1,+1],
                            [+1,-1,+1],
                            [+1,+1,+1],
                            [-1,+1,+1]  ])
            mesh.rel_nodes = np.asarray(mesh.rel_nodes, dtype=int)
            mesh.element_nodes = []
            for node in mesh.rel_nodes: 
                mesh.element_nodes.append( np.where( (mesh.nodes0==node[0]) & \
                    (mesh.nodes1==node[1]) & (mesh.nodes2==node[2]) )[0][0] + 1 )
            mesh.elements[ct_elements,:] = np.asarray( mesh.element_nodes )
            ct_elements += 1
#--------------------------------------------------------------------------------------------------
# create grains
#--------------------------------------------------------------------------------------------------
mesh.grain_seeds = []
for z in range(0, dim.edge_z - dim.grain_z + 1, dim.grain_z):
    for y in range(0, dim.edge_y - dim.grain_y + 1, dim.grain_y):
        for x in range(0, dim.edge_x - dim.grain_x + 1, dim.grain_x):
            mesh.grain_seeds.append( z * dim.edge_x*dim.edge_y + y * dim.edge_x + x + 1 )
            # TODO check above line for generality
dim.num_grains = len(mesh.grain_seeds)
mesh.grain_els = {}
for n, grain_seed in enumerate(mesh.grain_seeds): 
    mesh.grain_list = []
    for z in range(0,dim.grain_z):  # 0 to G-1, incl.
        # WARNING:  below only good for case of G=2, right? 
        for y in range(0,dim.grain_y):
            for x in range(0,dim.grain_x):
                mesh.grain_list += [ grain_seed + z * dim.edge_x*dim.edge_y + y * dim.edge_x + x ]
    mesh.grain_els[n] = mesh.grain_list
#--------------------------------------------------------------------------------------------------
# material sections
with open(files.sections, 'a') as f:
    for n in range(len(mesh.grain_seeds)):
        f.write('*Solid Section, elset=Grain' + str(n+1) + 
                '_set, material=Grain' + str(n+1) + '_Phase1_mat\n')
    f.write('**')
#--------------------------------------------------------------------------------------------------
# element sets
with open(files.elset,'w+') as f:
    f.write(
        '** Defines element sets\n'
        '*Elset, elset=cube, generate\n'
        '1, ' + str(dim.num_elements) + ', 1\n')
    for i in range(dim.num_grains):
        grain_list = mesh.grain_els[i]
        if i != 0: f.write('\n')
        f.write('*Elset, elset=Grain' + str(i+1) + '_set\n')
        for n in range(len(grain_list)): # TODO make this a function, is used later 
            if n == len(grain_list)-1:
                f.write(str(grain_list[n]))
            elif (n+1) % 16 == 0 and n !=0:
                f.write(str(grain_list[n]) + ',\n')
            else:
                f.write(str(grain_list[n]) + ', ')
        if i == dim.num_grains:
            f.write('**')
#--------------------------------------------------------------------------------------------------
# write out nodes, elements
with open(files.nodes, 'w+') as f:  # TODO lowercase filenames
    f.write('*NODE, NSET=ALLNODES\n')
    for i in range(dim.num_nodes - 1):
        f.write(str(i+1) + ', ' + ', '.join( separate(mesh.nodes[i,:])) + '\n')
    f.write(str(dim.num_nodes) + ', ' + ', '.join( separate(mesh.nodes[-1,:]) ))

with open(files.elements, 'w+') as f:
    f.write('*ELEMENT, TYPE=C3D8, ELSET=ALLELEMENTS\n')
    for i in range(dim.num_elements - 1):
        f.write(str(i+1) + ', ' + ', '.join( separate(mesh.elements[i,:])) + '\n')
    f.write(str(dim.num_elements)  + ', ' +  ', '.join( separate(mesh.elements[-1,:])))
#--------------------------------------------------------------------------------------------------
# nodesets
#--------------------------------------------------------------------------------------------------
# mesh.right = mesh.left = mesh.top = mesh.bottom = mesh.front = mesh.back = []
mesh.nset_list = [[] for _ in range(6)]
mesh.nset_names = ['RIGHT', 'LEFT', 'TOP', 'BOTTOM', 'FRONT', 'BACK']
for i, node in enumerate(mesh.nodes):
    if node[0] == dim.edge_x:   mesh.nset_list[0].append(int(i+1))
    elif node[0] == 0:          mesh.nset_list[1].append(int(i+1))
    if node[1] == dim.edge_y:   mesh.nset_list[2].append(int(i+1))
    elif node[1] == 0:          mesh.nset_list[3].append(int(i+1))
    if node[2] == dim.edge_z:   mesh.nset_list[4].append(int(i+1))
    elif node[2] == 0:          mesh.nset_list[5].append(int(i+1))
    # WARNING some nodes in multiple nodesets
with open(files.nodeset, 'w+') as f:
    f.write('**Defines node sets')
    for i, nset in enumerate(mesh.nset_list):
        f.write('\n*Nset, nset=' + mesh.nset_names[i] + '\n')
        for n in range(len(nset)):
            if n == len(nset)-1:
                f.write(str(nset[n]))
            elif (n+1) % 16 == 0 and n !=0:
                f.write(str(nset[n]) + ',\n')
            else:
                f.write(str(nset[n]) + ', ')
#--------------------------------------------------------------------------------------------------
# orientation:
#--------------------------------------------------------------------------------------------------
# function to get vectors:
def rand_orient(orient_obj):
    def random_miller(n):
        vector = []
        for _ in range(n):
            vector.append( randrange(-(orient_obj.max_index+1), orient_obj.max_index+1) )
        return vector
    x =y = 3*[0]
    while y == 3*[0]:
        y = random_miller(3)
    if y[2]==0:
        x[0] = x[1] = 0
        x[2] = 1
    else:
        x = [ *random_miller(2), 0]
        x[2] = np.round(((x[0]*y[0] + x[1]*y[1])/(-y[2])),decimals=5)
    orient_obj.assign( x, y )
# write vectors to file:
with open(files.orients,'w+') as f:
            f.write('*Parameter \n')
            orient_obj = orient()
            for a in range(1, dim.num_grains + 1):
                rand_orient(orient_obj)
                f.write('**Local direction of the global x direction \n')
                f.write('x' + str(a) + ' = ' + str(orient_obj.x[0]) + '\n')
                f.write('y' + str(a) + ' = ' + str(orient_obj.x[1]) + '\n')
                f.write('z' + str(a) + ' = ' + str(orient_obj.x[2]) + '\n')
                f.write('**Local direction of the global y direction \n')
                f.write('u' + str(a) + ' = ' + str(orient_obj.y[0]) + '\n')
                f.write('v' + str(a) + ' = ' + str(orient_obj.y[1]) + '\n')
                f.write('w' + str(a) + ' = ' + str(orient_obj.y[2]) + '\n')
                f.write('** -------------------------------------------------')
                if a != dim.num_grains:
                    f.write('\n')
#--------------------------------------------------------------------------------------------------
# write all other files
#--------------------------------------------------------------------------------------------------
# parameters
with open(files.mesh_extra, 'w') as f:
    dim.ref_node = int( np.ceil(dim.num_nodes/1e7) * 1e7 )
    # ^ start ref numbering at next 10 millionth node for clear separation from regular nodes
    f.write("""** mesh parameters
*Parameter
xmax = """ + str(dim.edge_x) + """
ymax = """ + str(dim.edge_y) + """
zmax = """ + str(dim.edge_z) + """
xmin = """ + str(0) + """
ymin = """ + str(0) + """
zmin = """ + str(0) + """
x_Half = (xmax-xmin)/2
y_Half = (ymax-ymin)/2
z_Half = (zmax-zmin)/2
*Node
""" + str(dim.ref_node + 0) + """,    <x_Half>,      <y_Half>,          <zmin>
""" + str(dim.ref_node + 1) + """,    <x_Half>,      <y_Half>,          <zmax>
""" + str(dim.ref_node + 2) + """,    <x_Half>,        <ymax>,        <z_Half>
""" + str(dim.ref_node + 3) + """,      <xmax>,      <y_Half>,        <z_Half>
""" + str(dim.ref_node + 4) + """,      <xmin>,      <y_Half>,        <z_Half>
""" + str(dim.ref_node + 5) + """,    <x_Half>,        <ymin>,        <z_Half>
*Nset, nset=RP-Back
""" + str(dim.ref_node + 0) + """
*Nset, nset=RP-Front
""" + str(dim.ref_node + 1) + """
*Nset, nset=RP-Top
""" + str(dim.ref_node + 2) + """
*Nset, nset=RP-Right
""" + str(dim.ref_node + 3) + """
*Nset, nset=RP-Left
""" + str(dim.ref_node + 4) + """
*Nset, nset=RP-Bottom
""" + str(dim.ref_node + 5) + """
**""")
#--------------------------------------------------------------------------------------------------
# main input file
with open(files.main,'w') as f:
    f.write("""** main input file
*include, input=""" + files.elements + """
*include, input=""" + files.elset + """
*include, input=""" + files.nodes + """
*include, input=""" + files.nodeset + """
*include, input=""" + files.mesh_extra + """
*include, input=""" + files.sections + """
*include, input=""" + files.orients + """
*include, input=""" + files.material + """
*include, input=""" + files.slip + """
**
*Equation
2
Top , 2, -1
RP-Top, 2,  1
2
Bottom , 2, -1
RP-Bottom, 2,  1
2
Front , 3, -1
RP-Front, 3,  1
2
Back , 3, -1
RP-Back, 3,  1
2
Left , 1, -1
RP-Left, 1,  1
2
Right , 1, -1
RP-Right, 1,  1
**
***RESTART,WRITE,FREQUENCY=5
*STEP, name=Loading, INC=1000000, NLGEOM, unsymm=YES, extrapolation=NO
*STATIC
1E-8,1.0,1E-9,0.005
*Boundary
RP-Bottom, 2, 2
RP-Back,   3, 3
RP-Left,   1, 1
RP-Top, 2, 2, """ + str(dim.disp) + """
**
*Output, field, Number interval=30, Time Marks=No
*Node output
RF, U
*Element Output, directions=YES
LE, PE, PEEQ, S, SDV
*END STEP""")
#--------------------------------------------------------------------------------------------------
# material definition
with open(files.material, 'w') as f:
    f.write("""** Material hardening parameters in the Bassani-Wu model
*Parameter
** Elastic Moduli
** Unit: MPa
C11 = 265.e3
C12 = 161.e3
C44 = 127.e3
** Constitutive relation (power law)
Gamma0 = 0.001
** Unit: s^-1
n = 50.
** should be larger than 1 (n = 1/m)
** Hardening law (hyperbolic function)
** Unit: MPa
Tau0 = 12.92
H0   = 40.8
TauS = 40
hs = 0.01
gamma0 = 0.4
f0 = 1
q = 1
**Second slip system family info:
gamma1 = 1
f1 = 1
q1 = 1""")
#--------------------------------------------------------------------------------------------------
# material properties for each grain
with open(files.slip, 'w+') as f:
    f.write('** Material properties for each grain')
mesh.grain_lines = []
for n in range(dim.num_grains):
    mesh.grain_lines.append( """
** ----------------------------------------------------------------------------
*Material, name = Grain""" + str(n+1) + """_Phase1_mat
*Depvar
125
*User material, Constants=160, unsymm
    <C11> ,  <C12> ,  <C44> ,
    0.   ,
    0.   ,
    1.   ,
    1.   ,   1.   ,   1.   ,   1.   ,   1.   ,   0.   ,
    0.   ,
    0.   ,
    <x"""+str(n+1)+""">   ,  <y"""+str(n+1)+""">,    <z"""+str(n+1)+""">,        1,       0,       0,
    <u"""+str(n+1)+""">   ,  <v"""+str(n+1)+""">,    <w"""+str(n+1)+""">,        0,       1,       0,
    <n>  ,<Gamma0>  ,
    0.   ,   0.
    0.   ,   0.   ,
    <H0>  , <TauS> , <Tau0>,  <hs>,  <gamma0>,  <gamma1>,  <f0>,  <f1> 
    <q>   ,  <q1>   ,
    0.   ,
    0.   ,
    0.   ,
    0.   ,
    .5   ,   1.   ,
    1.   ,   10.  , 1.E-5  ,""")
# write the rest of the file
with open(files.slip, 'a') as f:
    for line in range(len(mesh.grain_lines)):
        f.write(str(mesh.grain_lines[line]))
