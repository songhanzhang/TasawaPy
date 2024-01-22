import numpy as np
import matplotlib.pyplot as plt
def plt_elements(Nodes,Elements):
    n_elements = Elements.shape[1]
    for i_e in range(n_elements):
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
        plt.fill(x_profile, y_profile, facecolor = "whitesmoke", edgecolor = "gray", linewidth = 1)

def plt_nodes(Nodes):
    plt.scatter(Nodes[:,1], Nodes[:,2], s = 15, c = "dodgerblue", marker = "o")

def gen_list_DOF(n_nodes,n_dir):
    n_DOF = length(n_dir_list) * n_nodes
    list_DOF = zeros(n_DOF,2)
    counter = 0
    for i_node in range(n_nodes):
        for i_dir in range(n_dir_list):
            counter += 1
            list_DOF[counter,1] = counter
            list_DOF[counter,2] = i_node + 0.1*n_dir_list[i_dir]
    return n_DOF, list_DOF