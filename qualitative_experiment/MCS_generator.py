import torch
import matplotlib.cm as cm
from torch.utils.data import TensorDataset
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj, to_networkx, subgraph
import networkx as nx
import numpy as np

"""
Finding maximum common subgraph
Implemented by Eunbi Yoon
"""


class MCS_generation():
    def __init__(self, model, G1, G2):
        self.model = model
        self.G1 = G1
        self.G2 = G2
        self.V1 = G1.x.shape[0]  # num
        self.V2 = G2.x.shape[0]  # num
        self.A1 = to_dense_adj(self.G1.edge_index)[0]
        self.A2 = to_dense_adj(self.G2.edge_index)[0]
        self.Y = self.matching_matrix_generation()

    def get_U(self, G, mode='test'):  # G : torch.geometric  data
        self.model(G, training_with_batch=False)
        U = self.model.entire_embs
        return U

    def label_mask(self):
        x1 = self.G1.x
        x2 = self.G2.x
        x2T = torch.transpose(x2, 0, 1)
        return torch.matmul(x1, x2T)  # same element

    def matching_matrix_generation(self):
        U1 = self.get_U(self.G1)
        U2 = self.get_U(self.G2)
        U2T = torch.transpose(U2, 0, 1)
        X = torch.matmul(U1, U2T)  # num_G1 nodes * num _G2 nodes
        M = torch.max(X)
        self.U1 = U1
        self.U2 = U2
        exp_X = torch.exp(X - M)
        partition_col = exp_X.sum(1, keepdim=True) + 1e-5
        partition_row = exp_X.sum(0, keepdim=True) + 1e-5
        p_col = exp_X / partition_col * torch.sigmoid(X.sum(1, keepdim=True) / self.V2)
        p_low = exp_X / partition_row * torch.sigmoid(X.sum(0, keepdim=True) / self.V1)

        p = (p_col + p_low) / 2
        Y = p * self.label_mask()
        return Y

    def get_index(self, n):
        v1 = n // self.V2
        v2 = n % self.V2
        return v1, v2

    def subgraph_finding(self, max_epochs=1000, order=1,
                         decay=1.5, threshold=0.001):
        node1 = []
        node2 = []

        if order == 1:
            v1, v2 = self.initial_pair()

        else:
            v1, v2 = self.nth_pair(order)
        node1.append(v1)
        node2.append(v2)

        for epoch in range(1, 1 + max_epochs):

            NB1, NB2 = self.find_nbd(v1, v2)

            for i in range(min(len(NB1), len(NB2))):

                if True:
                    v1 = NB1[i]
                    v2 = NB2[i]
                    if self.Y[v1, v2] < threshold:
                        break
                    node1.append(v1)
                    node2.append(v2)
                    self.Y[v1, v2] = self.Y[v1, v2] / decay
                    break

        return node1, node2

    def select_subgraph(self, max_index=5, max_epochs=1000, decay=1.5, threshold=0.001):
        sizes = []
        node1s = []
        node2s = []
        for m in range(1, max_index + 1):
            node1, node2 = self.subgraph_finding(order=m, max_epochs=max_epochs,
                                                 decay=decay, threshold=threshold)
            l1 = len(np.unique(np.array(node1)))
            l2 = len(np.unique(np.array(node2)))
            node1s.append(node1)
            node2s.append(node2)
            size = min(l1, l2)
            sizes.append(size)
        sizes = np.array(sizes)
        M = np.argmax(sizes)
        node1 = node1s[M]
        node2 = node2s[M]
        return node1, node2

    def initial_pair(self):
        Y = self.Y
        index = torch.argmax(Y)
        v1, v2 = self.get_index(index)

        feature_i = torch.where(self.G1.x[v1] == 1)[0]
        feature_j = torch.where(self.G2.x[v2] == 1)[0]

        while feature_i != feature_j:
            Y[v1, v2] = 0
            self.initial_pair()
        v1 = int(v1)
        v2 = int(v2)
        return v1, v2

    def nth_pair(self, n):
        Y = self.Y
        i, j = self.initial_pair()
        Y[i, j] = 0
        Z = Y.view(-1)
        I = torch.argsort(Z, dim=0, descending=True)
        I = self.get_index(I)
        I = TensorDataset(I[0], I[1])
        v1 = int(I[n - 2][0])
        v2 = int(I[n - 2][1])
        feature_i = torch.where(self.G1.x[v1] == 1)[0]
        feature_j = torch.where(self.G2.x[v2] == 1)[0]
        return v1, v2

    def find_nbd(self, v1, v2):
        a1 = self.A1[v1].reshape(-1, 1)
        nbd1 = torch.where(a1 == 1)[0]

        a2 = self.A2[v2].reshape(-1, 1)
        nbd2 = torch.where(a2 == 1)[0]

        Y = self.Y
        B = Y.view(-1)
        C = torch.argsort(B, dim=0, descending=True)
        C = self.get_index(C)
        C = TensorDataset(C[0], C[1])

        NB1 = []
        NB2 = []

        for index in C:
            i = int(index[0])
            j = int(index[1])
            feature_i = torch.where(self.G1.x[i] == 1)[0]
            feature_j = torch.where(self.G2.x[j] == 1)[0]
            if i not in nbd1:
                continue
            if j not in nbd2:
                continue
            if feature_i != feature_j:
                continue
            NB1.append(i)
            NB2.append(j)

        if len(NB1) == 0 or len(NB2) == 0:
            for index in C:
                i = int(index[0])
                j = int(index[1])
                if i not in nbd1:
                    continue
                if j not in nbd2:
                    continue
                NB1.append(i)
                NB2.append(j)

        return NB1, NB2


def draw_subgraph(G, node, node_index, color_list, name=None, **kwds):
    edge_index = subgraph(node, G.edge_index)[0]
    data = Data(G.x, edge_index)
    S = to_networkx(data, to_undirected=True)
    index = []
    for i, node in enumerate(S.nodes):
        if node in nx.isolates(S):
            continue
        index.append(i)
    S.remove_nodes_from(list(nx.isolates(S)))
    node_feature = torch.where(data.x[index] == 1)[1]
    node_feature_name = [node_index[int(i)] for i in node_feature]
    labels = {i: name for i, name in enumerate(node_feature_name)}
    colors = [color_list[int(i)] for i in node_feature]
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_title(f'  {name} class {int(G.y)} subgraph')
    pos = nx.kamada_kawai_layout(S)
    color = node_feature
    nx.draw_networkx(S, pos=pos, with_labels=True, node_color=colors, **kwds)