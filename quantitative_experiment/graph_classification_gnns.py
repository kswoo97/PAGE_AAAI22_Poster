"""
GNN Models
"""

import torch
from torch_geometric.nn import GCNConv, SAGEConv

class GCN_Conv(torch.nn.Module):
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[8, 8], device = "cpu"):
        super(GCN_Conv, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.device = device
        self.last_dim = latent_dim[-1]

        self.convs.append(
            gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, (len(latent_dim) - 1)):
            self.convs.append(
                gconv(latent_dim[i], latent_dim[i + 1])
            )
        self.last_linear = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self):
        # Reset linear / convolutional parameters
        for conv_layer in self.convs:
            conv_layer.reset_parameters()
        self.last_linear.reset_parameters()

    def forward(self, data, training_with_batch):
        if training_with_batch : # Batch training
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            outs = torch.zeros((1, self.last_dim), requires_grad=True).to(self.device)
            for i in range(data.batch.unique().shape[0]) :
                idx = torch.where(data.batch == i)[0]
                batch_sum = torch.sum(x[idx, :], 0)
                outs = torch.vstack([outs, batch_sum])
            outs = outs[1:, :]
            self.embs = outs
            x = self.last_linear(outs)
        else : # Not batchwise training
            x, edge_index= data.x, data.edge_index
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            self.entire_embs = x[:, :]
            x = torch.sum(x, 0)
            self.embs = x
            x = self.last_linear(x)
        return torch.sigmoid(x)

class SAGE_Conv(torch.nn.Module):
    def __init__(self, dataset, gconv=SAGEConv, latent_dim=[8, 8], device = "cpu"):
        super(SAGE_Conv, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.device = device
        self.last_dim = latent_dim[-1]

        self.convs.append(
            gconv(dataset.num_features, latent_dim[0]))
        for i in range(0, (len(latent_dim) - 1)):
            self.convs.append(
                gconv(latent_dim[i], latent_dim[i + 1])
            )
        self.last_linear = torch.nn.Linear(latent_dim[-1], 1)

    def reset_parameters(self):
        # Reset linear / convolutional parameters
        for conv_layer in self.convs:
            conv_layer.reset_parameters()
        self.last_linear.reset_parameters()

    def forward(self, data, training_with_batch):
        if training_with_batch : # Batch training
            x, edge_index = data.x, data.edge_index
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            outs = torch.zeros((1, self.last_dim), requires_grad=True).to(self.device)
            for i in range(data.batch.unique().shape[0]) :
                idx = torch.where(data.batch == i)[0]
                batch_sum = torch.sum(x[idx, :], 0)
                outs = torch.vstack([outs, batch_sum])
            outs = outs[1:, :]
            self.embs = outs
            x = self.last_linear(outs)
        else : # Not batchwise training
            x, edge_index= data.x, data.edge_index
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            self.entire_embs = x[:, :]
            x = torch.sum(x, 0)
            self.embs = x
            x = self.last_linear(x)
        return torch.sigmoid(x)