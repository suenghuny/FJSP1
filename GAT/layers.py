import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, teleport_probability, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.teleport_probability = teleport_probability

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.c = nn.Parameter(torch.empty(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.c.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def edge_index_into_adjacency_matrix(self, n_node_feature, edge_index):
        adjacency_matrix = torch.zeros([n_node_feature, n_node_feature])
        len_edge_index = len(edge_index[0])
        for e in range(len_edge_index):
            i = edge_index[0][e]
            j = edge_index[1][e]
            adjacency_matrix[i][j] = 1
        return adjacency_matrix

    def forward(self, h, edge_index, n_node_features, mini_batch = False):

        if mini_batch == False:
            adj = torch.sparse_coo_tensor(edge_index,
                                           torch.ones(torch.tensor(edge_index).shape[1]).to(device),
                                           (n_node_features, n_node_features)).to_dense().to(device)

            adj = adj.to(device).long()
            Wh = torch.mm(h, self.W)                                                    # adj.shape : (n_node, n_node)

                                                                                        # h.shape : (n_node, feature_size), self.W.shape : (feature_size, hidden_size)
                                                                                        # Wh.shape : (n_node, hidden_size)
            e = self._prepare_attentional_mechanism_input(Wh, mini_batch = mini_batch)  # e : attention 계수(h_i 계산시 v_i와 연결된 v_j에 대해 얼마나 가중치를 둘 것 인가에 대한 계수)
            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(adj > 0, e, zero_vec)                               # e(attention 계수) 중에 연결성이 없는 것은 0처리
                                                                                        # e.shape : (n_node, n_node)
            attention = F.softmax(attention, dim=1)                                     # attention : (n_node, n_node)
            #attention = F.dropout(attention, self.dropout, training=self.training)
                                                        # Wh.shape : (n_node, hidden_size)
            h_prime =self.teleport_probability * torch.mm(attention, Wh) + (1-self.teleport_probability) * Wh

        else:
            #torch.sparse_coo_tensor(A[b][i][0], A[b][i][1], (num_nodes, num_nodes)).to(device).to_dense()
            #print(torch.tensor(edge_index[0]).shape)
            #torch.sparse_coo_tensor(edge_index[m], torch.ones(edge_index.shape[1]), (n_node_features, n_node_features)).to(device).to_dense()
            batch_size = len(edge_index)
            # adj = [
            #     self.edge_index_into_adjacency_matrix(n_node_features, edge_index[m]) for m in range(batch_size)]
            # print(torch.tensor(edge_index[0]).shape, torch.ones(len(edge_index[0][0])).shape)
            # print(torch.tensor(edge_index[3]).shape, torch.ones(len(edge_index[0][3])))

            W = self.W.expand([batch_size, self.in_features, self.out_features])
            Wh = torch.bmm(h, W)  # h.shape: (1, in_features), Wh.shape: (in_features, out_features)
            e = self._prepare_attentional_mechanism_input(Wh, mini_batch = mini_batch)  # e.shape: 1, num_enemy+1
            zero_vec = -9e15 * torch.ones_like(e)




            adj = [torch.sparse_coo_tensor(edge_index[m],
                                           torch.ones(torch.tensor(edge_index[m]).shape[1]).to(device),
                                           (n_node_features, n_node_features)).to_dense().to(device) for m in range(batch_size)]

            adj = torch.stack(adj)
            adj = adj.to(device).long()

            attention = torch.where(adj > 0, e, zero_vec)  # adj.shape: 1, num_enemy

            attention = F.softmax(attention, dim=2)

            h_prime = self.teleport_probability * torch.bmm(attention, Wh) + (1-self.teleport_probability) * Wh
            #print(attention_duration, adj1_duration)


        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh, mini_batch):
        if mini_batch == False:
            Wh1 = torch.mm(Wh, self.a[:self.out_features, :])       # Wh.shape      : (n_node, hidden_size), self.a : (hidden_size, 1)
            Wh2 = torch.mm(Wh, self.a[self.out_features:, :])       # Wh1 & 2.shape : (n_node, 1)
            e = Wh1 + Wh2.T                                         # e.shape       : (n_node, n_node)
        else:
            batch_size = Wh.shape[0]                                # Wh.shape      : (batch_size, n_node, out_feature)
            a = self.a.expand([batch_size, 2*self.out_features, 1]) # a.shape       : (batch_size, out_features, 1)
            Wh1 = torch.bmm(Wh, a[:, :self.out_features, :])        # Wh1 & 2.shape : (batch_size, n_node, 1)
            Wh2 = torch.bmm(Wh, a[:, self.out_features:, :])
            e = Wh1 + Wh2.view([batch_size, 1, -1])                 # e.shape       : (batch_size, n_node, n_node)


        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'






