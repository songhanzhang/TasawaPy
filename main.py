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

# plt.figure(figsize = (6.4,6.4))
fig, ax = plt.subplots(figsize = (6.4,6.4))
ax.tick_params(direction = 'in')
plt_elements(Nodes,Elements)
plt_nodes(Nodes)
plt.axis("equal")
plt.savefig("fig_model.pdf")
plt.show()

np.savez('model.npz', Nodes = Nodes, Elements = Elements, Materials = Materials, Reals = Reals)
model = np.load('model.npz')
Nodes = model['Nodes']
Elements = model['Elements']
Materials = model['Materials']
Reals = model['Reals']
