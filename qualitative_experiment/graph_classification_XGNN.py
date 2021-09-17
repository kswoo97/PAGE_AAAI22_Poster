import numpy as np
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

"""
XGNN
Original paper : Hao Yuan et al. 'XGNN: Towards Model-Level Explanations of Graph Neural Networks'. KDD 20.
Implemented by : Yonsei App.Stat. Sunwoo Kim
"""


class XGNN_model(torch.nn.Module):
    def __init__(self, dataset, candidates, data_type,
                 gconv=GCNConv, gcn_latent_dim=[16, 16], starting_mlp=16, ending_mlp=24,  # Given
                 device="cpu"):
        super(XGNN_model, self).__init__()
        self.candidates = candidates  # Given as a whole feature vector
        self.device = device
        self.data_type = data_type
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(in_channels=dataset.x.shape[1],
                                out_channels=gcn_latent_dim[0]))
        for i in range(len(gcn_latent_dim) - 1):
            self.convs.append(gconv(in_channels=gcn_latent_dim[i],
                                    out_channels=gcn_latent_dim[i + 1]))

        self.first_mlp = torch.nn.ModuleList()
        self.second_mlp = torch.nn.ModuleList()

        self.first_mlp.append(torch.nn.Linear(gcn_latent_dim[-1], starting_mlp))
        self.first_mlp.append(torch.nn.Linear(starting_mlp, 1))

        self.second_mlp.append(torch.nn.Linear(2 * gcn_latent_dim[-1], ending_mlp))
        self.second_mlp.append(torch.nn.Linear(ending_mlp, 1))

    def reset_parameters(self):
        for convs in self.convs:
            convs.reset_parameters()
        for mlp1 in self.first_mlp:
            mlp1.reset_parameters()
        for mlp2 in self.second_mlp:
            mlp2.reset_parameters()

    def forward(self, data):
        # Where to be started
        starting_mask = torch.vstack([torch.ones((data.x.shape[0], 1), device=self.device),
                                      torch.zeros((self.candidates.shape[0], 1), device=self.device)])
        # Where to be ended
        x, edge_index = torch.vstack([data.x, self.candidates]), data.edge_index
        first_x = x[:, :]
        ending_mask = torch.ones((x.shape[0], 1), device=self.device)

        # First convolutional layer
        for one_convs in self.convs:
            x = torch.relu(one_convs(x=x, edge_index=edge_index))
        node_feat = x[:, :]

        # First MLP layer
        x = torch.relu(self.first_mlp[0](x))
        x = self.first_mlp[1](x)
        out1 = x[:, :]
        x = torch.softmax(x, dim=0)
        x = x * starting_mask  # putting mask on (N+N' x 1)
        starting_idx = torch.argmax(x)  # Choose the starting node
        starting_feat = node_feat[starting_idx, :]
        node_feat = torch.hstack([node_feat,
                                  starting_feat.repeat(node_feat.shape[0], 1)])
        x = node_feat

        # Second MLP layer
        x = torch.relu(self.second_mlp[0](x))
        x = self.second_mlp[1](x)
        out2 = x[:, :]
        x = torch.softmax(x, dim=0)
        ending_mask[starting_idx, :] = 0
        x = x * ending_mask
        ending_idx = torch.argmax(x)
        if ending_idx >= data.x.shape[0]:
            new_features = first_x[ending_idx, :]
            adding_type = "new_node"
        else:
            new_features = None
            adding_type = "old_node"
        return starting_idx, ending_idx, adding_type, new_features, out1, out2

def data_generator(current_data, start_node, ending_node, adding_type):
    if adding_type == "new_node":  # Adding  a new node
        new_x = current_data.x.shape[0]
        new_edge_index = torch.vstack([torch.hstack([current_data.edge_index[0], torch.tensor([start_node, new_x])]),
                                       torch.hstack([current_data.edge_index[1], torch.tensor([new_x, start_node])])])
        X = torch.vstack([current_data.x, ending_node])
    else:
        new_edge_index = torch.vstack(
            [torch.hstack([current_data.edge_index[0], torch.tensor([start_node, ending_node])]),
             torch.hstack([current_data.edge_index[1], torch.tensor([ending_node, start_node])])])
        X = current_data.x
    return Data(x=X, edge_index=new_edge_index)

def rewarding_tf(gnn_model, new_g, label):
    gnn_model.eval()
    yhat = gnn_model(new_g, training_with_batch=False)
    if label == 1 :
        resulting_reward = yhat -  0.5
    elif label == 0 :
        resulting_reward = 0.5 - yhat
    else :
        raise TypeError("Label should be given one of 1 or 0")
    return resulting_reward

def total_reward(gnn_model, explainer, new_g, m, lambda_1, adding_type, label):
    Rtf = rewarding_tf(gnn_model, new_g, label = label).to("cpu").item()
    roll_reward = []
    for i in range(m):
        start_ind, ending_node, adding_type, added_feat, y1, y2 = explainer(new_g)
        if adding_type == "new_node":
            future_d = data_generator(new_g, start_ind, added_feat, adding_type = adding_type)
        else:  # Existing
            future_d = data_generator(new_g, start_ind, ending_node, adding_type = adding_type)
        future_reward = rewarding_tf(gnn_model, future_d, label = label)
        roll_reward.append(future_reward.to("cpu").item())
        new_g = future_d
        # visualizer(new_g)
    rewarding = Rtf + lambda_1 * (sum(roll_reward) / m)
    return rewarding

def generate_initial_graph(initial_n, data_type, init_type = 0) :

    if data_type == "ba_house" :
        if init_type == 0 :
            G = nx.barabasi_albert_graph(n=initial_n, m=2, seed=3000)
            edges = np.array(G.edges)
            total_edges = np.hstack([edges.transpose(), np.array([edges[:, 1], edges[:, 0]])])
            i_n = initial_n
            imp_edges = np.array(
                [[0, i_n, (i_n - 1), i_n, i_n, i_n + 1, i_n, i_n+2, i_n+1, i_n+2],
                 [i_n, 0, i_n, (i_n - 1), i_n + 1, i_n, i_n+2, i_n, i_n+2, i_n+1]])
            edge_index = torch.tensor(np.hstack([total_edges, imp_edges]), dtype=torch.long)
            X = torch.zeros((i_n+3, 3))
            X[:i_n, 2] = 1
            X[i_n, 0] = 1
            X[i_n+1:, 1] = 1
            initial_graph = Data(x = X, edge_index = edge_index)

        elif init_type == 1 :
            G = nx.barabasi_albert_graph(n=initial_n, m=2, seed=3000)
            edges = np.array(G.edges)
            total_edges = np.hstack([edges.transpose(), np.array([edges[:, 1], edges[:, 0]])])
            i_n = initial_n
            imp_edges = np.array(
                [[0, i_n, (i_n - 1), i_n, i_n, i_n + 1, i_n, i_n+2, i_n+1, i_n+2, 0, i_n+3, i_n+3, i_n+4, i_n+3, i_n+5],
                 [i_n, 0, i_n, (i_n - 1), i_n + 1, i_n, i_n+2, i_n, i_n+2, i_n+1, i_n+3, 0, i_n+4, i_n+3, i_n+5, i_n+3]])
            edge_index = torch.tensor(np.hstack([total_edges, imp_edges]), dtype=torch.long)
            X = torch.zeros((i_n+6, 3))
            X[:i_n, 2] = 1
            X[i_n, 0] = 1
            X[i_n + 1, 1] = 1
            X[i_n + 2, 1] = 1
            X[i_n + 3, 0] = 1
            X[i_n + 4, 1] = 1
            X[i_n + 5, 1] = 1
            initial_graph = Data(x = X, edge_index = edge_index)

        elif init_type == 2 :
            G = nx.barabasi_albert_graph(n=initial_n, m=2, seed=3000)
            edges = np.array(G.edges)
            total_edges = np.hstack([edges.transpose(), np.array([edges[:, 1], edges[:, 0]])])
            i_n = initial_n
            imp_edges = np.array(
                [[0, i_n],
                 [i_n, 0]])
            edge_index = torch.tensor(np.hstack([total_edges, imp_edges]), dtype=torch.long)
            X = torch.zeros((i_n+1, 3))
            X[:i_n, 2] = 1
            X[i_n, 0] = 1
            initial_graph = Data(x = X, edge_index = edge_index)

        elif init_type == 3 :
            G = nx.barabasi_albert_graph(n=initial_n, m=2, seed=3000)
            edges = np.array(G.edges)
            total_edges = np.hstack([edges.transpose(), np.array([edges[:, 1], edges[:, 0]])])
            i_n = initial_n
            imp_edges = np.array(
                [[0, i_n, i_n, (i_n+1)],
                 [i_n, 0, (i_n+1), i_n]])
            edge_index = torch.tensor(np.hstack([total_edges, imp_edges]), dtype=torch.long)
            X = torch.zeros((i_n+2, 3))
            X[:i_n, 2] = 1
            X[i_n, 0] = 1
            X[(i_n+1), 1] = 1
            initial_graph = Data(x = X, edge_index = edge_index)

        else :
            G = nx.barabasi_albert_graph(n=initial_n, m=2, seed=3000)
            edges = np.array(G.edges)
            total_edges = np.hstack([edges.transpose(), np.array([edges[:, 1], edges[:, 0]])])
            i_n = initial_n
            imp_edges = np.array(
                [[0, i_n, i_n, (i_n+1), i_n, (i_n+2)],
                 [i_n, 0, (i_n+1), i_n, (i_n+2), i_n]])
            edge_index = torch.tensor(np.hstack([total_edges, imp_edges]), dtype=torch.long)
            X = torch.zeros((i_n+3, 3))
            X[:i_n, 2] = 1
            X[i_n, 0] = 1
            X[(i_n+1), 1] = 1
            X[(i_n+2), 1] = 1
            initial_graph = Data(x = X, edge_index = edge_index)

    elif data_type == "solubility" :
        if init_type == 0: # Long carbon ring
            X = torch.zeros((6, 9))
            X[:, 1] = 1
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                                       [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype = torch.long)
            initial_graph = Data(x = X, edge_index = edge_index)

        elif init_type == 1 : # Short carbon ring
            X = torch.zeros((2, 9))
            X[:, 1] = 1
            edge_index = torch.tensor([[0, 1],
                                       [1, 0]], dtype=torch.long)
            initial_graph = Data(x = X, edge_index = edge_index)
        elif init_type == 2 : # NO2
            X = torch.zeros((3, 9))
            X[0, 5], X[1, 6], X[2, 6]  = 1, 1, 1
            edge_index = torch.tensor([[0, 1, 0, 2],
                                       [1, 0, 2, 0]], dtype = torch.long)
            initial_graph = Data(x = X, edge_index = edge_index)

        elif init_type == 3 : # CO2
            X = torch.zeros((3, 9))
            X[0, 1], X[1, 6], X[2, 6] = 1, 1, 1
            edge_index = torch.tensor([[0, 1, 0, 2],
                                       [1, 0, 2, 0]], dtype=torch.long)
            initial_graph = Data(x=X, edge_index=edge_index)

        elif init_type == 4 : # NC2
            X = torch.zeros((4, 9))
            X[0, 5], X[1:, 1], = 1, 1,
            edge_index = torch.tensor([[0, 1, 0, 2, 2, 3],
                                       [1, 0, 2, 0, 0, 2]], dtype=torch.long)
            initial_graph = Data(x=X, edge_index=edge_index)
    return initial_graph


def visualizing(torch_data, data_type, title = "Resulting Graph"):
    """
    :param G: torch_geometric type data
    :param title: title of the plots
    :return: graph figure
    """
    plt.figure(figsize=(8, 6))
    if data_type == "ba_house":
        node_colors = []
        node_size = []
        edge_size = []
        G = nx.Graph()

        for i in range(torch_data.x.shape[0]):
            G.add_node(i)
            if torch.argmax(torch_data.x[i, :]) == 0:
                node_colors.append("red")
            elif torch.argmax(torch_data.x[i, :]) == 1:
                node_colors.append("lime")
            else:
                node_colors.append("orange")
            node_size.append(500)

        for s in range(torch_data.edge_index.shape[1]):
            G.add_edge(torch_data.edge_index[0, s].item(),
                       torch_data.edge_index[1, s].item())
            edge_size.append(2)
        nx.draw(G, node_color=node_colors, node_size = node_size,
               width = edge_size)
        plt.title(title)
        plt.show()
        return G

    elif data_type == "solubility":
        node_colors = []
        node_size = []
        edge_size = []
        G = nx.Graph()
        current_labels = {}

        for i in range(torch_data.x.shape[0]):
            G.add_node(i)
            node_size.append(500)
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
            else:
                raise TypeError("None of atom was searched")

        for s in range(torch_data.edge_index.shape[1]):
            G.add_edge(torch_data.edge_index[0, s].item(),
                       torch_data.edge_index[1, s].item())
            edge_size.append(2)
        pos = nx.kamada_kawai_layout(G)
        nx.draw_networkx_nodes(G, pos, nodelist=list(np.array(G.nodes)), node_color=node_colors, node_size = node_size)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges, width = edge_size)
        nx.draw_networkx_labels(G, pos, current_labels, font_size=15)
        plt.title(title)
        plt.axis('off')
        plt.show()
        return G

def train_XGNN(explainer, gnn_model, initial_n, max_node_n, lambda_1, lambda_2, m, label, init_type = 0, show_init_g = False) :
    """
    :param explainer: Declared XGNN model
    :param gnn_model: GNN model which we want to explain
    :param initial_n: Number of initial meaningless nodes
    :param max_node_n: Maximum number of nodes that are accepted for the explained graph
    :param lambda_1: Weight of future reward / Rollout (Usually integer)
    :param lambda_2: Weight of Rule-based graph (Usually integer)
    :param m: Exploration for the future (Steps to move further)
    :param label: Class we want to generate
    :return: Trained model and resulting graph
    :return:
    """

    data_type = explainer.data_type
    gnn_model.eval()
    exp_optim = torch.optim.Adam(explainer.parameters(), lr=0.01)
    Gt = generate_initial_graph(initial_n = initial_n, data_type = data_type, init_type = init_type)
    max_node_n = max_node_n
    lambda_2 = lambda_2
    ce_loss = torch.nn.CrossEntropyLoss()
    explainer.train()
    if show_init_g :
        visualizing(Gt, data_type = data_type, title = "Initial graph")
    print(gnn_model(Gt, training_with_batch = False))

    for step in range(1000):
        exp_optim.zero_grad()
        starting_node, ending_node, adding_type, new_feat, y1, y2 = explainer(Gt)
        y1 = y1.reshape(1, y1.shape[0], y1.shape[1])
        y2 = y2.reshape(1, y2.shape[0], y2.shape[1])
        if adding_type == "new_node":
            new_Gt = data_generator(Gt, starting_node, new_feat, adding_type)
        else:
            new_Gt = data_generator(Gt, starting_node, ending_node, adding_type)
        Rt = total_reward(gnn_model=gnn_model,
                          explainer=explainer,
                          new_g=new_Gt,
                          m=m,
                          lambda_1=lambda_1,
                          adding_type=adding_type,
                          label = label)
        if new_Gt.x.shape[0] > max_node_n:
            Rtr = -1
        else:
            Rtr = 0
        reward = Rt + lambda_2 * Rtr
        loss = -reward * (ce_loss(y1, torch.tensor([starting_node]).unsqueeze(0))
                          + ce_loss(y2, torch.tensor([ending_node]).unsqueeze(0)))
        loss.backward()
        exp_optim.step()
        if reward > 0:
            Gt = new_Gt
    visualizing(Gt, data_type = data_type, title = "Generated graph")
    return explainer, Gt