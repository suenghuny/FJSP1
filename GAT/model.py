import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT.layers import GraphAttentionLayer, device

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, teleport_probability, mode = 'observation'):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.mode = mode
        self.teleport_probability = teleport_probability


        if mode == 'communication':
            self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, teleport_probability=self.teleport_probability).to(device)
                                for _ in range(nheads)]
            for i, attention in enumerate(self.attentions1):
                self.add_module('attention_{}'.format(i), attention)
            self.attentions2 = [GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True, teleport_probability=self.teleport_probability).to(device) for _ in
                range(nheads)]

            for i, attention in enumerate(self.attentions2):
                self.add_module('attention_{}'.format(i), attention)

            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, teleport_probability=self.teleport_probability).to(
                device)


        if mode == 'observation':
            self.attentions1 = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True, teleport_probability=self.teleport_probability).to(device)
                                for _ in range(nheads)]
            for i, attention in enumerate(self.attentions1):
                self.add_module('attention_{}'.format(i), attention)
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False, teleport_probability=self.teleport_probability).to(
                device)

    #
    # def save(self):
    #     if self.mode == 'communication':
    #         s
    #     if self.mode == 'observation':

    def forward(self, x, edge_index, n_node_features, mini_batch):

        #x = F.dropout(x, self.dropout, training=self.training)
        if mini_batch == False:
            if self.mode == 'communication':
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions1], dim=1)
                x = F.elu(x)
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions2], dim=1)
                x = F.elu(self.out_att(x, edge_index, n_node_features, mini_batch))
            if self.mode == 'observation':
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions1], dim=1)
                x = F.elu(self.out_att(x, edge_index, n_node_features, mini_batch))
        else:
            if self.mode == 'communication':
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions1], dim=2)
                x = F.elu(x)
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions2], dim=2)
                x = F.elu(self.out_att(x, edge_index, n_node_features, mini_batch))
            if self.mode == 'observation':
                x = torch.cat([att(x, edge_index, n_node_features, mini_batch) for att in self.attentions1], dim=2)
                x = F.elu(self.out_att(x, edge_index, n_node_features, mini_batch))
        return x

