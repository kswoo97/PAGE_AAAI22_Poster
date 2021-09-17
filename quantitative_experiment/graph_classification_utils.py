import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from pysmiles import read_smiles
import networkx as nx
import torch.nn.functional as F

"""
Finding house shape data
Yonsei App.Stat. Sunwoo Kim
There are only three features
"""

"""
Generate data
"""


class ba_house_generator() :

    def __init__(self, max_n, min_n, edge_rate, r_seed) :

        self.max_n = max_n
        self.min_n = min_n
        self.edge_rate = edge_rate
        self.data_type = "single_house" # Will be changed soon
        self.seed = r_seed
        self.nx_list = []
        self.data_list = []
        self.data_type_list = []

    def full_false_example(self, seed, node_n) :
        G = nx.barabasi_albert_graph(n=node_n, m=self.edge_rate, seed=seed)
        G.add_edge(int(node_n - 1), int(node_n))
        G.add_edge(int(node_n - 1), int(node_n + 1))
        G.add_edge(int(node_n), int(node_n + 1))
        G.add_edge(int(node_n), int(node_n + 2))
        G.add_edge(int(node_n + 1), int(node_n + 3))
        G.add_edge(int(node_n + 2), int(node_n + 3))
        np.random.seed(seed)
        entire_set = [[int(node_n - 1), int(node_n)],
                                     [int(node_n - 1), int(node_n + 1)], [int(node_n), int(node_n + 1)],
                                     [int(node_n), int(node_n + 2)], [int(node_n + 1), int(node_n + 3)],
                                     [int(node_n + 2), int(node_n + 3)]]
        removing = np.random.choice([0, 1, 2, 3, 4, 5], size = 1, p = [0.15, 0.15, 0.25, 0.15, 0.15, 0.15])[0]
        G.remove_edge(entire_set[removing][0], entire_set[removing][1])
        np.random.seed(seed)
        G.add_edge(0, int(node_n + 4))  # In order to avoid model's trick, we add one more red edge
        G.add_edge(int(node_n), int(node_n + 4)) # In order to avoid model's trick, we add one more red edge
        return G

    def random_generator(self, y, seed) :
        '''
        [Top  Middle  Bottom  None]
        :param data_type:
        :param y:
        :param seed:
        :return:
        '''

        np.random.seed(seed)
        node_n = int(np.random.choice(np.arange(self.min_n, self.max_n), 1))
        s = 0
        if y == 0 : # Build full house
            np.random.seed(seed)
            new_k = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
            if self.data_type == "single_house" :
                G = nx.barabasi_albert_graph(n=node_n, m = self.edge_rate, seed=seed)
                G.add_edge(int(node_n-1), int(node_n))
                G.add_edge(int(node_n-1), int(node_n+1))
                G.add_edge(int(node_n), int(node_n+1))
                G.add_edge(int(node_n), int(node_n+2))
                G.add_edge(int(node_n+1), int(node_n+3))
                G.add_edge(int(node_n+2), int(node_n+3))
                if new_k >= 1 :
                    for tt in range(new_k) :
                        np.random.seed(seed + tt)
                        G.add_edge(np.random.choice(range(node_n-1), 1)[0],
                                   np.random.choice(range(node_n, node_n+3), 1)[0])
                        if new_k < 4 :
                            G.add_edge(int(tt), int(node_n + 4 + tt))

                edges = np.array(G.edges)
                inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                       edges[:, 0].reshape(-1, 1)]) # Changing order to tell its undirected
                edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                          dtype = torch.long)
                x = torch.zeros((int(np.array(G.nodes).shape[0]), 3))
                x[:int(node_n-1), 2] = 1 # Nothing
                x[int(node_n-1), 0] = 1 # Head
                x[int(node_n), 1] = 1 # Middle
                x[int(node_n + 1), 1] = 1 # Middle
                x[int(node_n + 2), 1] = 1 # Bottom
                x[int(node_n + 3), 1] = 1 # Bottom
                x[int(node_n) + 4:, 1] = 1  # Bottom
                self.data_type_list.append("single_house")

            else : # Double house

                G = nx.barabasi_albert_graph(n=node_n, m=self.edge_rate, seed=seed)
                # First house (n-2, n, n+1, n+2, n+3)
                G.add_edge(int(node_n - 2), int(node_n))
                G.add_edge(int(node_n - 2), int(node_n + 1))
                G.add_edge(int(node_n), int(node_n + 1))
                G.add_edge(int(node_n), int(node_n + 2))
                G.add_edge(int(node_n + 1), int(node_n + 3))
                G.add_edge(int(node_n + 2), int(node_n + 3))

                # Second house (n-1, n+4, n+5, n+6, n+7)
                G.add_edge(int(node_n - 1), int(node_n + 4))
                G.add_edge(int(node_n - 1), int(node_n + 5))
                G.add_edge(int(node_n + 4), int(node_n + 5))
                G.add_edge(int(node_n + 4), int(node_n + 6))
                G.add_edge(int(node_n + 5), int(node_n + 7))
                G.add_edge(int(node_n + 6), int(node_n + 7))

                G.add_edge(0, int(node_n-2))
                G.add_edge(0, int(node_n-1))
                G.add_edge(int(node_n - 1), int(node_n - 2))

                if new_k >= 1 :
                    for tt in range(new_k) :
                        np.random.seed(seed + tt)
                        if new_k < 4:
                            G.add_edge(int(tt), int(node_n + 8 + tt))

                edges = np.array(G.edges)
                inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                       edges[:, 0].reshape(-1, 1)])  # Changing order to tell its undirected
                edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                          dtype=torch.long)
                x = torch.zeros((np.array(G.nodes).shape[0], 3))
                x[:int(node_n - 2), 2] = 1  # Nothing
                x[int(node_n - 2), 0] = 1  # Head
                x[int(node_n - 1), 0] = 1  # Head
                x[int(node_n), 1] = 1  # Middle
                x[int(node_n + 1), 1] = 1  # Middle
                x[int(node_n + 2), 1] = 1  # Bottom
                x[int(node_n + 3), 1] = 1  # Bottom
                x[int(node_n + 4), 1] = 1  # Middle
                x[int(node_n + 5), 1] = 1  # Middle
                x[int(node_n + 6), 1] = 1  # Bottom
                x[int(node_n + 7), 1] = 1  # Bottom
                x[int(node_n)+8:, 1] = 1
                self.data_type_list.append("double_houses")

            data = Data(x = x, edge_index = edge_index.t().contiguous(), y = torch.tensor([1], dtype = torch.float))

        elif y == 1 : # Build a partial house
            np.random.seed(seed)
            s = int(np.random.choice([1, 2, 3, 4], 1, p = [0.2, 0.2, 0.2, 0.4]))

            if s == 4 :
                np.random.seed(seed)
                G = self.full_false_example(seed = seed, node_n=node_n)

            elif s == 3 :
                G = nx.barabasi_albert_graph(n=(node_n + s), m=self.edge_rate, seed=seed)
                np.random.seed(seed)
                new_k = np.random.choice([0, 1, 2, 3], 1)[0]
                if new_k > 0 :
                    for i in range(new_k) :
                        G.add_edge((node_n + s - 1), (node_n + s - 1 + i))

            else :
                G = nx.barabasi_albert_graph(n=(node_n+s), m=self.edge_rate, seed=seed)

            edges = np.array(G.edges)
            inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                   edges[:, 0].reshape(-1, 1)])  # Changing order to tell its undirected
            edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                      dtype=torch.long)

            x = torch.zeros((np.array(G.nodes).shape[0], 3))
            x[:int(node_n - 1), 2] = 1  # Nothing
            x[int(node_n - 1), 0] = 1  # Head
            if s == 1 :
                x[int(node_n), 1] = 1  # Middle
            elif s == 2 :
                x[int(node_n), 1] = 1
                x[int(node_n+1), 1] = 1
            elif s == 3: # se == 3 :
                x[int(node_n), 1] = 1
                x[int(node_n) : , 1] = 1
            else : # s == 4
                x[int(node_n), 1] = 1
                x[int(node_n + 1), 1] = 1
                x[int(node_n + 2), 1] = 1
                x[int(node_n + 3), 1] = 1
                x[int(node_n + 4), 0] = 1
            data = Data(x=x, edge_index=edge_index.t().contiguous(), y = torch.tensor([0], dtype = torch.float))
            self.data_type_list.append("partial_house")

        elif y == 2 : # Do not build a house
            G = nx.barabasi_albert_graph(n=node_n, m=self.edge_rate, seed=seed)
            # Randomly put some branches to the graph
            np.random.seed(seed)
            num_branch = np.random.choice([0, 1, 2, 3, 4], 1)[0]
            for i in range(num_branch) :
                np.random.seed(seed + i)
                mother_node = np.random.choice([0, 1, 2, 3, 4], 1)[0]
                G.add_edge(mother_node, node_n + i)
            edges = np.array(G.edges)
            inv_edges = np.hstack([edges[:, 1].reshape(-1, 1),
                                   edges[:, 0].reshape(-1, 1)])  # Changing order to tell its undirected
            edge_index = torch.tensor(np.vstack([edges, inv_edges]),
                                      dtype=torch.long)
            x = torch.zeros((np.array(G.nodes).shape[0], 3))
            np.random.seed(seed)
            random_one = np.random.choice(range(np.array(G.nodes).shape[0] - 1), 1)[0]
            x[:int(node_n - 1), 2] = 1  # Nothing
            x[int(node_n - 1), 0] = 1  # Head
            x[int(node_n):, 1] = 1  # branches
            x[int(random_one), 0] = 1
            data = Data(x=x, edge_index=edge_index.t().contiguous(), y = torch.tensor([0], dtype = torch.float))
            self.data_type_list.append("null_house")

        else :
            raise TypeError("For single house example, y should be given one of {0, 1, 2}")

        self.nx_list.append([G, [y, s], self.data_type])
        self.data_list.append(data)

    def visualization(self, n):
        color_map = []
        node_n = self.data_list[n].x.shape[0]
        G = self.nx_list[n][0]
        for i in range(int(node_n)):
            if torch.argmax(self.data_list[n].x[i, :]) == 0 :
                color_map.append("red")
            elif torch.argmax(self.data_list[n].x[i, :]) == 1:
                color_map.append("lime")
            else :
                color_map.append("orange")
        nx.draw(G, node_color=color_map, with_labels=True, pos=nx.kamada_kawai_layout(G))
        plt.show()

    def dataset_generator(self, num_graph) :
        seed = self.seed
        for epoch in range(num_graph) :
            np.random.seed(seed); cur_i = np.random.choice([0, 1, 2], 1)
            np.random.seed(seed); self.data_type = np.random.choice(["single_house",
                                                                     "double_house"])
            if cur_i == 0 : # Generate Full house
                self.random_generator(y = 0, seed = seed)
            elif cur_i == 1 : # Generate partial house
                self.random_generator(y = 1, seed=seed)
            else : # Generate null-house
                self.random_generator(y = 2, seed=seed)
            seed += 1

def solubility_data_generator(path) :
    raw_data = pd.read_csv(path)

    smile_codes = raw_data.smiles.values
    ys = raw_data.iloc[:, 8].values

    raw_data["y"] = raw_data.iloc[:, 8]
    raw_data["target"] = raw_data["y"].apply(lambda x: 1 if x > -2 else (0 if x < -4 else -1))
    raw_data = raw_data[raw_data.target > -0.5].reset_index().iloc[:, 1:]

    smile_codes = raw_data.smiles.values
    ys = raw_data.iloc[:, 8].values

    new_y = raw_data.target.values

    chemical_indexs = {"Br": 0,
                       "C": 1,
                       "Cl": 2,
                       "F": 3,
                       "I": 4,
                       "N": 5,
                       "O": 6,
                       "P": 7,
                       "S": 8}
    ## There are 9 types of chemical indexs. Thus molecular feature should have 9-dimensional vectors.
    ## There are [1.0, 1.5, 2.0, 3.0] types of bond. Thus edge attribute has 4-dimensional vectors,

    data = []
    for i in range(new_y.shape[0]) :
        tmp_y = new_y[i]
        formula = smile_codes[i]
        mol = read_smiles(formula)
        node_x = torch.zeros((len(nx.get_node_attributes(mol, name="element")),
                              9), dtype=torch.float)
        edge_indexs = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attrs = torch.zeros((1, 4), dtype=torch.long)
        for x_ in range(node_x.shape[0]):
            item_ = nx.get_node_attributes(mol, name="element")[x_]
            node_x[x_, chemical_indexs[item_]] = 1

        ## Edge type 1.0
        index1 = torch.tensor([np.where(nx.to_numpy_matrix(mol, weight='order') == 1.0)[0],
                               np.where(nx.to_numpy_matrix(mol, weight='order') == 1.0)[1]], dtype=torch.long)
        if index1.shape[1] > 0:
            x1 = torch.zeros((index1.shape[1], 4), dtype=torch.float)
            x1[:, 1] = 1
            edge_indexs = torch.hstack([edge_indexs, index1])
            edge_attrs = torch.vstack([edge_attrs, x1])

        ## Edge type 1.5
        index1 = torch.tensor([np.where(nx.to_numpy_matrix(mol, weight='order') == 1.5)[0],
                               np.where(nx.to_numpy_matrix(mol, weight='order') == 1.5)[1]], dtype=torch.long)
        if index1.shape[1] > 0:
            x1 = torch.zeros((index1.shape[1], 4), dtype=torch.float)
            x1[:, 1] = 1
            edge_indexs = torch.hstack([edge_indexs, index1])
            edge_attrs = torch.vstack([edge_attrs, x1])

        ## Edge type 2.0
        index1 = torch.tensor([np.where(nx.to_numpy_matrix(mol, weight='order') == 2.0)[0],
                               np.where(nx.to_numpy_matrix(mol, weight='order') == 2.0)[1]], dtype=torch.long)
        if index1.shape[1] > 0:
            x1 = torch.zeros((index1.shape[1], 4), dtype=torch.float)
            x1[:, 1] = 1
            edge_indexs = torch.hstack([edge_indexs, index1])
            edge_attrs = torch.vstack([edge_attrs, x1])

        ## Edge type 3.0
        index1 = torch.tensor([np.where(nx.to_numpy_matrix(mol, weight='order') == 3.0)[0],
                               np.where(nx.to_numpy_matrix(mol, weight='order') == 3.0)[1]], dtype=torch.long)
        if index1.shape[1] > 0:
            x1 = torch.zeros((index1.shape[1], 4), dtype=torch.float)
            x1[:, 1] = 1
            edge_indexs = torch.hstack([edge_indexs, index1])
            edge_attrs = torch.vstack([edge_attrs, x1])

        tmp_data = Data(x=node_x,
                        edge_index=edge_indexs[:, 1:],
                        edge_attr=edge_attrs[1:, :],
                        y=torch.tensor([tmp_y], dtype=torch.float))

        data.append(tmp_data)
    return data

# Training utility functions

def ba_house_class_evaluator(GNN_model, test_list, device) :
    test_loss = 0
    GNN_model.eval()
    for part_d in test_list :
        part_d = part_d.to(device)
        y_ = GNN_model(part_d, training_with_batch = False)
        if y_ > 0.5 :
            if part_d.y > 0.5 :
                test_loss += 1
        else :
            if part_d.y <= 0.5 :
                test_loss += 1
    return test_loss/len(test_list)

def solubility_class_evaluator(GNN_model, test_list, device) :
    test_loss = 0
    GNN_model.eval()
    for part_d in test_list :
        part_d = part_d.to(device)
        y_ = GNN_model(part_d, training_with_batch = False)
        if y_ > 0.5 :
            if part_d.y > 0.5 :
                test_loss += 1
        else :
            if part_d.y <= 0.5 :
                test_loss += 1
    return test_loss/len(test_list)

def ba_house_training(model, data_module, test_data, device, epochs = 300) :
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    model = model.to(device)
    model.train()
    criterion = torch.nn.BCELoss().to(device)
    for epoch in range(epochs):
        for data in data_module :
            data.to(device)
            optimizer.zero_grad()
            out = model(data = data, training_with_batch = True)
            loss = criterion(out.view(-1), data.y)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 :
            test_result = ba_house_class_evaluator(GNN_model=model, test_list = test_data, device = device)
            if test_result >= 0.99 :
                break
    return model

def solubility_training(model, data_module, test_data, device, epochs = 300) :
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    model = model.to(device)
    criterion = torch.nn.BCELoss().to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for data in data_module:
            data.to(device)
            optimizer.zero_grad()
            out = model(data=data, training_with_batch=True)
            loss = criterion(out.view(-1), data.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.to("cpu").detach().item()
    return model