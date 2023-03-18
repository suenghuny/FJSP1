import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from GTN.inits import glorot, zeros



device = f'cuda' if torch.cuda.is_available() else 'cpu'

"""
 num_FastGTN_layers = 1,
 hidden_size = 64,
 num_channel = 3,
 num_FastGT_layers = 2
"""




class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, feature_size, num_nodes, num_FastGTN_layers, hidden_size, num_channels, num_layers, gtn_beta, teleport_probability):
        # w_in은 node_feature의 갯수
        super(FastGTNs, self).__init__()
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = num_FastGTN_layers
        self.gtn_beta = gtn_beta
        self.teleport_probability = teleport_probability
        fastGTNs = []
        for i in range(self.num_FastGTN_layers):
            if i == 0:
                fastGTNs.append(FastGTN(num_edge_type = num_edge_type, w_in = feature_size, w_out = hidden_size, num_nodes = num_nodes, num_channels = num_channels, num_layers = num_layers, gtn_beta = gtn_beta, teleport_probability= self.teleport_probability))
            else:
                fastGTNs.append(
                    FastGTN(num_edge_type=num_edge_type,
                            w_in=hidden_size,
                            w_out=hidden_size,
                            num_nodes=num_nodes,
                            num_channels=num_channels,
                            num_layers=num_layers,
                            gtn_beta = gtn_beta,
                            teleport_probability= self.teleport_probability))
        self.fastGTNs = nn.ModuleList(fastGTNs)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, A, X, num_nodes=None, mini_batch = False):
        """

        Parameters
        ----------
        A : edge_index(adjacency matrix) 및 edge_weight를 포함 : heterogeneous matrix의 개수 만큼 포함
        X : node_feature

        num_nodes
        eval
        args
        n_id
        node_labels
        epoch
        -------

        """

        if num_nodes == None:
            num_nodes = self.num_nodes
        """
        GNN layer의 층수를 의미
        """

        H_ = self.fastGTNs[0](A, X, num_nodes=num_nodes, mini_batch = mini_batch)

        for i in range(1, self.num_FastGTN_layers):
            H_ = self.fastGTNs[i](A, H_, num_nodes=num_nodes, mini_batch = mini_batch)
        return H_


class FastGTN(nn.Module):
    def __init__(self, num_edge_type, w_in, w_out, num_nodes, num_channels, num_layers, gtn_beta, teleport_probability):
        super(FastGTN, self).__init__()

        self.num_edge_type = num_edge_type
        self.num_channels = num_channels
        self.num_nodes = num_nodes  # w_in

        self.w_in = w_in
        self.w_out = w_out
        self.num_layers = num_layers
        self.gtn_beta = gtn_beta
        self.teleport_probability = teleport_probability

        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(FastGTLayer(num_edge_type = num_edge_type, num_channels = self.num_channels, num_nodes = num_nodes, first=True))
            else:
                layers.append(FastGTLayer(num_edge_type = num_edge_type, num_channels = self.num_channels, num_nodes = num_nodes, first=False))
        self.layers = nn.ModuleList(layers)

        self.Ws = []

        for i in range(self.num_channels):
            self.Ws.append(nn.Parameter(torch.Tensor(self.w_in, self.w_out)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)




    def forward(self, A, X, num_nodes, mini_batch):


        if mini_batch == False:
            H = [X @ W for W in self.Ws]  # GCNConv와 input의 matrix multiplication
            H = torch.stack(H, dim=0)
            X_ = H.clone().detach().requires_grad_(False)
            #print("전", X_.shape)
            for i in range(self.num_layers):
                # H가 모든 channel에 대한 X@W를 답고 있음
                # self.layers[i]는 GTLayer를 담고 있음
                H = self.layers[i](H, A, num_nodes, layer=i + 1,mini_batch = mini_batch)  # self.layers 내부에서 channel별로 연산이 수행됨(출력된 H의 길이는 channel 길이와 동일함)


            """
            이제 여기서부터 node feature를 이용한 graph convolution 연산이 수행된다.
            """

            H_ = self.teleport_probability * F.relu(self.gtn_beta * X_ + (1 - self.gtn_beta) * H)+(1-self.teleport_probability)* X_

            H_ = torch.einsum("ijk ->jik", H_)

            H_ = H_.reshape(num_nodes, -1)

            H_ = F.relu(self.linear1(H_))


        else:
            H = [torch.einsum('bij, jk->bik', X, W) for W in self.Ws]  # GCNConv와 input의 matrix multiplication
            H = torch.stack(H, dim=0)
            H = torch.einsum('cbik -> bcik', H)
            X_ = H.clone().detach().requires_grad_(True)


            """
            layer 단위 연산(논문상 K)
            """
            for i in range(self.num_layers):
                # H가 모든 channel에 대한 X@W를 답고 있음
                # self.layers[i]는 GTLayer를 담고 있음
                H = self.layers[i](H, A, num_nodes, layer=i + 1,mini_batch = mini_batch)  # self.layers 내부에서 channel별로 연산이 수행됨(출력된 H의 길이는 channel 길이와 동일함)


            """
            이제 여기서부터 node feature를 이용한 graph convolution 연산이 수행된다.
            """
            shape0, shape1, shape2, shape3 = H.shape
            # batch_size, num_channels, num_nodes, feature_size
            H_ = self.teleport_probability * F.relu(self.gtn_beta * (X_) + (1 - self.gtn_beta) * H)+(1-self.teleport_probability)*X_
            H_ = torch.einsum("bijk -> bjik", H_)
            H_ = H_.reshape(shape0, shape2, -1)
            H_ = F.relu(self.linear1(H_))
        # Equation (15) : aggregation

        return H_


class FastGTLayer(nn.Module):

    def __init__(self, num_edge_type, num_channels, num_nodes, first=True):
        super(FastGTLayer, self).__init__()
        self.num_edge_type = num_edge_type
        self.num_channels =  num_channels
        self.first = first
        self.num_nodes = num_nodes

        self.weight = nn.Parameter(torch.Tensor(num_channels, num_edge_type))

        self.conv1 = FastGTConv(num_edge_type = num_edge_type,
                                num_channels = num_channels,
                                num_nodes = num_nodes)
        self.monitors_before = list()
        self.monitors_after = list()


    def forward(self, H_, A, num_nodes, layer=None, mini_batch = False):
        """
        channel 별 연산:
        Adjacency matrix(mat_a)를 만들어냄(sparse tensor로 만듬)
        """



        # Equation (16)에서 Z와 Adjacency matrix와 convolution filter의 convex combination을 의미하는 부분
        # result_A의 길이는 channel의 길이와 동일함
        # W1은 filter에 해당함함
        # self.conv1은 A를 입력받아 new meta-path adjacency matrix를 출력 받는다.
        if mini_batch == False:
            weight = self.weight
            filter = F.softmax(weight, dim=1)
            num_channels = filter.shape[0]
            Hs = []
            for i in range(num_channels):
                for j, (edge_index, edge_value) in enumerate(A):
                    if j == 0:
                        total_edge_index = edge_index
                        total_edge_value = edge_value * filter[i][j]
                    else:
                        total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                        total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))

                mat_a = torch.sparse_coo_tensor(total_edge_index.detach(), total_edge_value, (num_nodes, num_nodes)).to(
                    total_edge_value.device)
                mat_a = mat_a.coalesce()
                H = torch.sparse.mm(mat_a, H_[i])
                Hs.append(H)
            Hs = torch.stack(Hs, dim=0)
        else:
            batch_size = len(A)
            weight = self.weight
            filter = F.softmax(weight, dim=1)

            import time


            start = time.time()
            mat_a = list()
            for b in range(batch_size):
                temp = list()
                for i in range(self.num_edge_type):
                    temp.append(torch.sparse_coo_tensor(A[b][i][0], A[b][i][1], (num_nodes, num_nodes)).to(device).to_dense())
                mat_a.append(torch.stack(temp, dim=0))
            mat_a = torch.stack(mat_a, dim=0)
            self.monitors_after.append(time.time() - start)



            Hs = torch.einsum('bcij, bcjk-> bcik', torch.einsum('bijk, ci  -> bcjk', mat_a, filter), H_)


        return Hs


class FastGTConv(nn.Module):

    def __init__(self, num_edge_type, num_channels, num_nodes):
        super(FastGTConv, self).__init__()
        self.num_edge_type = num_edge_type     # num_edge_type
        self.num_channels =  num_channels       # num_channel
        self.weight = nn.Parameter(torch.Tensor(num_channels, num_edge_type))
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.num_nodes = num_nodes
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.normal_(self.weight, std=0.1)


    def forward(self, A, num_nodes, mini_batch = False):
        if mini_batch == False:
            weight = self.weight
            filter = F.softmax(weight, dim=1)
            mat_a = [torch.sparse_coo_tensor(A[i][0], A[i][1], (num_nodes, num_nodes)).to(device).to_dense()
                     for i in range(len(A))]
            mat_a = torch.stack(mat_a, dim=0)
            adj = torch.einsum('ci,ijk->cjk', filter, mat_a)
        else:
            batch_size = len(A)
            weight = self.weight
            filter = F.softmax(weight, dim=1)
            mat_a = torch.stack([torch.stack([torch.sparse_coo_tensor(A[b][i][0], A[b][i][1], (num_nodes, num_nodes)).to(device).to_dense() for i in range(len(A[0]))], dim = 0) for b in range(batch_size)], dim = 0)
            adj = torch.einsum('bijk,ci->bcjk', mat_a, filter)

        return adj



