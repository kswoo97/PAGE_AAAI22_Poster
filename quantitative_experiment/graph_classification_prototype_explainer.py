from torch_geometric.data import Data
import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from MCS_generator import *
from itertools import combinations

"""
Generating prototypes in graph-level
Implemented by Yonsei App.Stat. Sunwoo Kim
Co-worked with Eunbi Yoon & Yongmin Shin
Advised by Prof. Wonyong Shin
"""

class prototype_explanation() :

    def __init__(self, gnn_model, dataset, data_name) :

        self.model = gnn_model
        self.data = dataset
        self.data_type = data_name
        self.n_components = 2

    def multiple_mahalanobis(self, x, means, prec):
        return np.diag(((x - means).dot(prec)).dot((x - means).transpose()))

    def shifted_graph_to_torch(self, G, features):
        node_trans = {}
        edge_index = torch.tensor([[0, 0]], dtype = torch.long)
        if self.data_type == "ba_house" :
            feature_vector = torch.zeros((1,3), dtype = torch.float)
            for i, j in enumerate(G.nodes) :
                node_trans[j] = i
                added_feat = torch.zeros((1,3), dtype = torch.float)
                added_feat[0, features[i]] = 1
                feature_vector = torch.vstack([feature_vector, added_feat])
            for n1, n2 in G.edges :
                added_edge = torch.tensor([[node_trans[n1], node_trans[n2]],
                                           [node_trans[n2], node_trans[n1]]], dtype = torch.long)
                edge_index = torch.vstack([edge_index, added_edge])
            feature_vector = feature_vector[1:, :]
            edge_index = edge_index[1:, :]
            data = Data(x = feature_vector, edge_index = edge_index.t().contiguous())
        elif self.data_type == "solubility" :
            feature_vector = torch.zeros((1, 9), dtype=torch.float)
            for i, j in enumerate(G.nodes):
                node_trans[j] = i
                added_feat = torch.zeros((1, 9), dtype=torch.float)
                added_feat[0, features[i]] = 1
                feature_vector = torch.vstack([feature_vector, added_feat])
            for n1, n2 in G.edges:
                added_edge = torch.tensor([[node_trans[n1], node_trans[n2]],
                                           [node_trans[n2], node_trans[n1]]], dtype=torch.long)
                edge_index = torch.vstack([edge_index, added_edge])
            feature_vector = feature_vector[1:, :]
            edge_index = edge_index[1:, :]
            data = Data(x=feature_vector, edge_index=edge_index.t().contiguous())
        else :
            raise TypeError("Datatype can be given one of ba_house or solubility")
        return data

    def visualizing(self, torch_data, title):
        """
        :param G: torch_geometric type data
        :param title: title of the plots
        :return: graph figure
        """

        if self.data_type == "ba_house" :
            node_colors = []
            G = nx.Graph()

            for i in range(torch_data.x.shape[0]):
                G.add_node(i)
                if torch.argmax(torch_data.x[i, :]) == 0:
                    node_colors.append("red")
                elif torch.argmax(torch_data.x[i, :]) == 1:
                    node_colors.append("lime")
                else:
                    node_colors.append("orange")

            for s in range(torch_data.edge_index.shape[1]):
                G.add_edge(torch_data.edge_index[0, s].item(),
                           torch_data.edge_index[1, s].item())
            nx.draw(G, node_color=node_colors, with_labels=True)
            plt.title(title)
            plt.show()
            return G

        elif self.data_type == "solubility" :
            plt.figure(figsize=(8, 6))
            node_colors = []
            G = nx.Graph()
            current_labels = {}

            for i in range(torch_data.x.shape[0]):
                G.add_node(i)
                if torch.argmax(torch_data.x[i, :]) == 0:
                    node_colors.append("red")
                    current_labels[i] = "Br"
                elif torch.argmax(torch_data.x[i, :]) == 1:
                    node_colors.append("pink")
                    current_labels[i] = "C"
                elif torch.argmax(torch_data.x[i, :]) == 2:
                    node_colors.append("orange")
                    current_labels[i] = "Cl"
                elif torch.argmax(torch_data.x[i, :]) == 3:
                    node_colors.append("lime")
                    current_labels[i] = "F"
                elif torch.argmax(torch_data.x[i, :]) == 4:
                    node_colors.append("green")
                    current_labels[i] = "I"
                elif torch.argmax(torch_data.x[i, :]) == 5:
                    node_colors.append("cyan")
                    current_labels[i] = "N"
                elif torch.argmax(torch_data.x[i, :]) == 6:
                    node_colors.append("blue")
                    current_labels[i] = "O"
                elif torch.argmax(torch_data.x[i, :]) == 7:
                    node_colors.append("purple")
                    current_labels[i] = "P"
                elif torch.argmax(torch_data.x[i, :]) == 8:
                    node_colors.append("grey")
                    current_labels[i] = "S"
                else :
                    raise TypeError("None of atom was searched")

            for s in range(torch_data.edge_index.shape[1]):
                G.add_edge(torch_data.edge_index[0, s].item(),
                           torch_data.edge_index[1, s].item())
            pos = nx.kamada_kawai_layout(G)
            nx.draw_networkx_nodes(G, pos, nodelist = list(np.array(G.nodes)), node_color = node_colors)
            nx.draw_networkx_edges(G, pos, edgelist=G.edges)
            nx.draw_networkx_labels(G, pos, current_labels, font_size=10)
            plt.title(title)
            plt.axis('off')
            plt.show()
            return G

    def generate_nearest_nodes(self, label, k, cluster_index) :
        """
        :param label: Which label(class) to be explained
        :param k: Number of nearest neighbors from the centroid
        :return: list of the nearest nodes
        """
        if cluster_index >= self.n_components :
            raise TypeError("Cluster index should be given smaller integer than number of components")
        ci = cluster_index
        self.model.eval()
        self.model(self.data[0], training_with_batch=False)
        embeddings = torch.empty((1, self.model.embs.shape[0]))
        ys = []
        for part_d in self.data :
            outs = self.model(part_d, training_with_batch = False)
            embeddings = torch.vstack([embeddings, self.model.embs])
            ys.append(part_d.y.item())
        entire_embs = embeddings[1:, :].detach().numpy()
        y1 = np.where(np.array(ys) == 1.0)[0]
        y0 = np.where(np.array(ys) == 0.0)[0]
        emb1 = entire_embs[y1, :]
        emb0 = entire_embs[y0, :]
        clus1 = GaussianMixture(n_components=self.n_components, random_state=0).fit(emb1)
        clus0 = GaussianMixture(n_components=self.n_components, random_state=0).fit(emb0)

        if label == 0 :
            dists = self.multiple_mahalanobis(emb0, clus0.means_[ci], clus0.precisions_[ci])
            sorting = np.sort(dists)
            top_nodes = []
            index_counting = 0
            while len(top_nodes) < k:
                index_counting += 1
                if sorting[int(index_counting - 1)] not in top_nodes:
                    top_nodes.append(sorting[int(index_counting - 1)])
                else:
                    continue
            pos = []
            for i in range(k):
                pos.append(y0[np.where(dists == top_nodes[i])[0]][0])
        elif label == 1 :
            dists = self.multiple_mahalanobis(emb1, clus1.means_[ci], clus1.precisions_[ci])
            sorting = np.sort(dists)
            top_nodes = []
            index_counting = 0
            while len(top_nodes) < k :
                index_counting += 1
                if sorting[int(index_counting-1)] not in top_nodes :
                    top_nodes.append(sorting[int(index_counting-1)])
                else :
                    continue
            pos = []
            for i in range(k) :
                pos.append(y1[np.where(dists == top_nodes[i])[0]][0])
        else :
            raise TypeError("Label should be given either 1 or 0")
        return pos

    def generate_prototype(self, label, k, n_components, n_iter, cluster_index, max_epochs = 40) :
        self.n_components = n_components
        pos = self.generate_nearest_nodes(label = label, k = k, cluster_index=cluster_index)
        combin = list(combinations(pos, 2))
        found_subgraphs = []
        for n1, n2 in combin :
            G1 = self.data[n1]
            G2 = self.data[n2]
            subg_length = []
            best_subg_searching = []
            self.for_testing = []

            for order in range(1, (n_iter+1)) :
                m = MCS_generation(self.model, G1=G1, G2=G2)
                node1, node2 = m.subgraph_finding(order = order, max_epochs = max_epochs)
                partial_edge_index = subgraph(node1, G1.edge_index)[0]
                data = Data(G1.x, partial_edge_index)
                self.for_testing.append(data)
                S = to_networkx(data, to_undirected=True)
                index = []
                for i, node in enumerate(S.nodes):
                    if node in nx.isolates(S):
                        continue
                    index.append(i)
                S.remove_nodes_from(list(nx.isolates(S)))
                node_feature = torch.where(data.x[index] == 1)[1]
                subg_length.append(node_feature.shape[0])
                best_subg_searching.append([S, node_feature])
            subg = best_subg_searching[np.argmax(subg_length)] # list shape
            found_subgraphs.append(self.shifted_graph_to_torch(
                G = subg[0], features = subg[1]))

        # We perform overall searching once again in order to subtract the final common subgraph
        self.leng = subg_length
        self.nn = found_subgraphs
        subg_n = range(len(combin))
        second_combin = list(combinations(subg_n, 2))
        final_graphs = []
        for n1, n2 in second_combin :
            G1 = found_subgraphs[n1]
            G2 = found_subgraphs[n2]
            m = MCS_generation(self.model, G1 = G1, G2 = G2)
            final_subg_length = []
            final_subg_searching = []

            for order in range(1, (n_iter+1)) :
                node1, node2 = m.subgraph_finding(order = order, max_epochs = max_epochs)
                partial_edge_index = subgraph(node1, G1.edge_index)[0]
                data = Data(G1.x, partial_edge_index)
                S = to_networkx(data, to_undirected=True)
                index = []
                for i, node in enumerate(S.nodes):
                    if node in nx.isolates(S):
                        continue
                    index.append(i)
                S.remove_nodes_from(list(nx.isolates(S)))
                node_feature = torch.where(data.x[index] == 1)[1]
                final_subg_length.append(node_feature.shape[0])
                final_subg_searching.append([S, node_feature])

            subg = final_subg_searching[np.argmax(final_subg_length)] # list shape
            final_graphs.append(self.shifted_graph_to_torch(
                G = subg[0], features = subg[1]))
        return final_graphs