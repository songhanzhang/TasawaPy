import numpy as np
import matplotlib.pyplot as plt
from Tasawa import *

work_path = '/Users/songhan.zhang/Documents/Julia/2023-Julia-v1205-SAFE/'

Nodes = np.array([ [ 1,  0,  0 ],
                   [ 2,  0.05,  0 ],
                   [ 3,  0.05*np.cos(np.pi/4),  0.05*np.sin(np.pi/4) ],
                   [ 4,  0.05*np.cos(2*np.pi/4),  0.05*np.sin(2*np.pi/4) ],
                   [ 5,  0.05*np.cos(3*np.pi/4),  0.05*np.sin(3*np.pi/4) ],
                   [ 6,  0.05*np.cos(4*np.pi/4),  0.05*np.sin(4*np.pi/4) ],
                   [ 7,  0.05*np.cos(5*np.pi/4),  0.05*np.sin(5*np.pi/4) ],
                   [ 8,  0.05*np.cos(6*np.pi/4),  0.05*np.sin(6*np.pi/4) ],
                   [ 9,  0.05*np.cos(7*np.pi/4),  0.05*np.sin(7*np.pi/4) ],
                   [ 10,  0.025,  0 ],
                   [ 11,  0,  0.025 ],
                   [ 12,  -0.025,  0 ],
                   [ 13,  0,  -0.025 ] ])
n_nodes = Nodes.shape[0]

Elements = np.array([ 
    [ 1,  "2D_QuadTriangle",  1,  1,  (1, 2, 4, 10, 3, 11) ],
    [ 2,  "2D_QuadTriangle",  1,  1,  (1, 4, 6, 11, 5, 12) ],
    [ 3,  "2D_QuadTriangle",  1,  1,  (1, 6, 8, 12, 7, 13) ],
    [ 4,  "2D_QuadTriangle",  1,  1,  (1, 8, 2, 13, 9, 10) ],
], dtype = object)
n_elements = Elements.shape[0]

Materials = np.array([ [ 1, (2e11, 7850, 0.3) ] ] , dtype = object)
Reals = np.array([ [ 1, (1) ] ], dtype = object)

plt.figure(figsize = (6.4,6.4))
plt.scatter(Nodes[:,1], Nodes[:,2], s = 15, c = "dodgerblue", marker = "o")

i_e = 1
node_1 = Elements[i_e-1,4][0]
node_2 = Elements[i_e-1,4][1]
node_3 = Elements[i_e-1,4][2]
node_4 = Elements[i_e-1,4][3]
node_5 = Elements[i_e-1,4][4]
node_6 = Elements[i_e-1,4][5]
x = np.zeros(6)
y = np.zeros(6)
x[0] = Nodes[node_1-1,1]
y[0] = Nodes[node_1-1,2]
x[1] = Nodes[node_2-1,1]
y[1] = Nodes[node_2-1,2]
x[2] = Nodes[node_3-1,1]
y[2] = Nodes[node_3-1,2]
x[3] = Nodes[node_4-1,1]
y[3] = Nodes[node_4-1,2]
x[4] = Nodes[node_5-1,1]
y[4] = Nodes[node_5-1,2]
x[5] = Nodes[node_6-1,1]
y[5] = Nodes[node_6-1,2]
coor_interp = np.array([
    [ 0.0,  0.0 ],
    [ 0.1,  0.0 ],
    [ 0.2,  0.0 ],
    [ 0.3,  0.0 ],
    [ 0.4,  0.0 ],
    [ 0.5,  0.0 ],
    [ 0.6,  0.0 ],
    [ 0.7,  0.0 ],
    [ 0.8,  0.0 ],
    [ 0.9,  0.0 ],
    [ 1.0,  0.0 ],
    [ 0.9,  0.1 ],
    [ 0.8,  0.2 ],
    [ 0.7,  0.3 ],
    [ 0.6,  0.4 ],
    [ 0.5,  0.5 ],
    [ 0.4,  0.6 ],
    [ 0.3,  0.7 ],
    [ 0.2,  0.8 ],
    [ 0.1,  0.9 ],
    [ 0.0,  1.0 ],
    [ 0.0,  0.9 ],
    [ 0.0,  0.8 ],
    [ 0.0,  0.7 ],
    [ 0.0,  0.6 ],
    [ 0.0,  0.5 ],
    [ 0.0,  0.4 ],
    [ 0.0,  0.3 ],
    [ 0.0,  0.2 ],
    [ 0.0,  0.1 ]
])
n_interp = coor_interp.shape[0]
x_profile = np.zeros(n_interp)
y_profile = np.zeros(n_interp)
for i_interp in range(n_interp):
    xi  = coor_interp[i_interp,0]
    eta = coor_interp[i_interp,1]
    Nb = np.zeros(6)
    Nb[0] = (1-xi-eta) * (1-2*xi-2*eta)
    Nb[1] = xi * (2*xi-1)
    Nb[2] = eta * (2*eta-1)
    Nb[3] = 4 * xi * (1-xi-eta)
    Nb[4] = 4 * xi * eta
    Nb[5] = 4 * eta * (1-xi-eta)
    x_profile[i_interp] = np.dot(Nb,x)
    y_profile[i_interp] = np.dot(Nb,y)

plt.axis("equal")
plt.show()