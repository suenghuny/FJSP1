


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy
from GTN.utils import _norm
from GTN.model_fastgtn import FastGTNs
from scipy.sparse import csr_matrix




class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim = 1)

class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q, action_size):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q+10)
        self.fcn_2 = nn.Linear(hidden_size_q+10, hidden_size_q-5)
        self.fcn_3 = nn.Linear(hidden_size_q-5, hidden_size_q-20)
        self.fcn_4 = nn.Linear(hidden_size_q-20, action_size)
        #self.fcn_5 = nn.Linear(int(hidden_size_q/8), action_size)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        #torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):

        x = F.relu(self.fcn_1(obs_and_action))
        x = F.relu(self.fcn_2(x))
        x = F.relu(self.fcn_3(x))
        q = self.fcn_4(x)
        #q = self.fcn_5(x)
        return q

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.fcn_1 = nn.Linear(feature_size, hidden_size+10)
        self.fcn_2 = nn.Linear(hidden_size+10, hidden_size+10)
        self.fcn_3 = nn.Linear(hidden_size+10, n_representation_obs)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)

    def forward(self, node_feature):
        x = F.relu(self.fcn_1(node_feature))
        x = F.relu(self.fcn_2(x))
        node_representation = self.fcn_3(x)
        return node_representation

class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent, action_size):
        self.buffer = deque()


        self.step_count_list = list()
        for _ in range(8):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.one_hot_actions = np.eye(action_size).tolist()
        self.batch_size = batch_size
        self.step_count = 0


    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature_machine, num_waiting_operations, edge_index_machine, action, reward, done, avail_action):
        #self.buffer[0].append(node_feature_job)
        self.buffer[1].append(node_feature_machine)
        self.buffer[2].append(num_waiting_operations)
        self.buffer[3].append(edge_index_machine)
        self.buffer[4].append(action)
        self.buffer[5].append(reward)
        self.buffer[6].append(done)
        self.buffer[7].append(avail_action)

        if self.step_count < self.buffer_size - 1:
            self.step_count_list.append(self.step_count)
            self.step_count += 1



    def generating_mini_batch(self, datas, batch_idx, cat):
        # self.buffer[0].append(node_feature_job)
        # self.buffer[1].append(node_feature_machine)
        # self.buffer[2].append(edge_index_job)
        # self.buffer[3].append(edge_index_machine)
        # self.buffer[4].append(action)
        # self.buffer[5].append(reward)
        # self.buffer[6].append(done)
        # self.buffer[7].append(avail_action)
        for s in batch_idx:
            # if cat == 'node_feature_job':
            #     yield datas[0][s]
            if cat == 'node_feature_machine':
                yield datas[1][s]
            if cat == 'num_waiting_operations':
                yield datas[2][s]
            if cat == 'edge_index_machine':
                yield datas[3][s]
            if cat == 'action':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]

            # if cat == 'node_feature_job_next':
            #     yield datas[0][s+1]
            if cat == 'node_feature_machine_next':
                yield datas[1][s+1]
            if cat == 'num_waiting_operations_next':
                yield datas[2][s+1]
            if cat == 'edge_index_machine_next':
                yield datas[3][s+1]
            if cat == 'avail_action_next':
                yield datas[7][s+1]



    def sample(self):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        sampled_batch_idx = random.sample(step_count_list, self.batch_size)

        # node_feature_job = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_job')
        # node_features_job = list(node_feature_job)

        node_feature_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_machine')
        node_features_machine = list(node_feature_machine)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')
        actions = list(action)


        num_waiting_operations = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='num_waiting_operations')
        num_waiting_operations = list(num_waiting_operations)

        num_waiting_operations_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='num_waiting_operations_next')
        num_waiting_operations_next = list(num_waiting_operations_next)




        edge_index_machine = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_machine')
        edge_indices_machine = list(edge_index_machine)



        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)



        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)


        #
        # node_feature_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_job_next')
        # node_features_job_next = list(node_feature_job_next)



        node_feature_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_machine_next')
        node_features_machine_next = list(node_feature_machine_next)



        # edge_index_job_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_job_next')
        # edge_indices_job_next = list(edge_index_job_next)



        edge_index_machine_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_machine_next')
        edge_indices_machine_next = list(edge_index_machine_next)

        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        return node_features_machine,  num_waiting_operations, edge_indices_machine, actions, rewards, dones,  node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next

class Agent:
    def __init__(self,
                 num_agent,
                 feature_size_job,
                 feature_size_machine,

                 hidden_size_obs,
                 hidden_size_comm,
                 hidden_size_Q,
                 hidden_size_meta_path,
                 n_multi_head,
                 n_representation_job,
                 n_representation_machine,

                 dropout,
                 action_size,
                 buffer_size,
                 batch_size,
                 learning_rate,
                 gamma,
                 GNN,
                 teleport_probability,
                 gtn_beta):
        torch.manual_seed(81)
        random.seed(81)
        np.random.seed(81)
        self.num_agent = num_agent
        self.feature_size_job = feature_size_job
        self.feature_size_machine = feature_size_machine
        self.hidden_size_meta_path = hidden_size_meta_path
        self.hidden_size_obs = hidden_size_obs
        self.hidden_size_comm = hidden_size_comm
        self.n_multi_head = n_multi_head
        self.teleport_probability = teleport_probability

        self.n_representation_job = n_representation_job
        self.n_representation_machine = n_representation_machine


        self.action_size = action_size

        self.dropout = dropout
        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()

        self.max_norm = 10
        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)



        self.VDN_target.load_state_dict(self.VDN.state_dict())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.num_agent, self.action_size)



        self.action_space = [i for i in range(self.action_size)]



        self.GNN = GNN
        if self.GNN == 'GAT':

            self.node_representation_job_obs = NodeEmbedding(feature_size=feature_size_job, hidden_size=hidden_size_obs,
                                                               n_representation_obs=n_representation_job).to(device)  # 수정사항


            self.node_representation = NodeEmbedding(feature_size=feature_size_machine, hidden_size=hidden_size_obs,
                                                     n_representation_obs=n_representation_machine).to(device)  # 수정사항

            self.func_job_obs = GAT(nfeat = n_representation_job,
                                      nhid = hidden_size_obs,
                                      nheads = n_multi_head,
                                      nclass = n_representation_job+10,
                                      dropout = dropout,
                                      alpha = 0.2,
                                      mode = 'observation',
                                    n_node = self.num_agent,
                                    batch_size=self.batch_size,
                                      teleport_probability = self.teleport_probability).to(device)


            self.func_machine_comm = GAT(nfeat = n_representation_machine,
                                      nhid = hidden_size_comm,
                                      nheads = n_multi_head,
                                      nclass = n_representation_machine+5,
                                      dropout = dropout,
                                      alpha = 0.2,
                                      mode = 'observation',
                                         n_node = self.num_agent,
                                         batch_size = self.batch_size,
                                      teleport_probability = self.teleport_probability).to(device)   # 수정사항



            self.Q = Network(n_representation_job+n_representation_machine+5, hidden_size_Q, self.action_size).to(device)
            self.Q_tar = Network(n_representation_job+n_representation_machine+5, hidden_size_Q, self.action_size).to(device)
            self.Q_tar.load_state_dict(self.Q.state_dict())

            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.node_representation_job_obs.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.func_job_obs.parameters()) + \
                               list(self.func_machine_comm.parameters())
        if self.GNN == 'FastGTN':
            self.Q = Network(
                hidden_size_meta_path +
                n_representation_obs,
                hidden_size_Q).to(device)
            self.Q_tar = Network(hidden_size_meta_path + n_representation_obs, hidden_size_Q).to(device)
            self.action_representation = NodeEmbedding(feature_size=feature_size + 6 - 1, hidden_size=hidden_size_obs,
                                                       n_representation_obs=n_representation_obs).to(device)  # 수정사항

            # num_edge_type, feature_size, num_nodes, num_FastGTN_layers, hidden_size, num_channels, num_FastGT_layers)
            self.func_meta_path = FastGTNs(num_edge_type=5,
                             feature_size=feature_size,
                             num_nodes=self.num_nodes,
                             num_FastGTN_layers = 2,
                             hidden_size = hidden_size_meta_path,
                             num_channels = 2,
                             num_layers = 2,
                             teleport_probability=self.teleport_probability,
                             gtn_beta = gtn_beta
                             ).to(device)
            self.node_representation = NodeEmbedding(feature_size=feature_size - 1, hidden_size=hidden_size_obs,
                                                     n_representation_obs=n_representation_obs).to(device)  # 수정사항
            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.func_meta_path.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.action_representation.parameters())
        self.optimizer = optim.RMSprop(self.eval_params, lr=learning_rate)
        self.time_check =[[],[]]




    def save_model(self, e, t, epsilon, path):
        torch.save({
            'e': e,
            't': t,
            'epsilon': epsilon,
            'Q': self.Q.state_dict(),
            'Q_tar': self.Q_tar.state_dict(),
            'node_representation_job_obs' : self.node_representation_job_obs.state_dict(),
            'node_representation' : self.node_representation.state_dict(),
            'func_job_obs': self.func_job_obs.state_dict(),
            'func_machine_comm': self.func_machine_comm.state_dict(),
            'VDN': self.VDN.state_dict(),
            'VDN_target': self.VDN_target.state_dict(),
            'optimizer' : self.optimizer.state_dict()}, "{}".format(path))


    def load_model(self, path):
        self = torch.load(path)



    def get_node_representation(self, node_feature_machine, num_waiting_operations, edge_index_machine, n_node_features_machine, mini_batch = False):
        if self.GNN == 'GAT':
            if mini_batch == False:
                with torch.no_grad():
                    node_feature_machine = torch.tensor(node_feature_machine,
                                                        dtype=torch.float,
                                                        device=device)

                    num_waiting_operations = torch.tensor(num_waiting_operations,
                                                        dtype=torch.float,
                                                        device=device)

                    node_embedding_num_waiting_operations = self.node_representation_job_obs(num_waiting_operations)
                    len_num_waiting_operations = node_embedding_num_waiting_operations.shape[0]

                    node_embedding_num_waiting_operations = node_embedding_num_waiting_operations.unsqueeze(0).expand([n_node_features_machine, len_num_waiting_operations])





#                    obs_n = obs.unsqueeze(1).expand([self.batch_size, self.action_size, self.n_representation_comm])

                    node_embedding_machine_obs = self.node_representation(node_feature_machine)
                    edge_index_machine = torch.tensor(edge_index_machine, dtype=torch.long, device=device)
                    node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine, n_node_features_machine, mini_batch = mini_batch)
                    #print(node_embedding_num_waiting_operations.shape, node_representation.shape)
                    node_representation = torch.cat([node_embedding_num_waiting_operations, node_representation], dim = 1)
            else:
                #node_feature_job = torch.tensor(node_feature_job, dtype=torch.float, device=device)
                node_feature_machine = torch.tensor(node_feature_machine, dtype=torch.float, device=device)
                num_waiting_operations = torch.tensor(num_waiting_operations,
                                                      dtype=torch.float,
                                                      device=device)

                node_embedding_num_waiting_operations = self.node_representation_job_obs(num_waiting_operations)
                len_num_waiting_operations = node_embedding_num_waiting_operations.shape[1]
                node_embedding_num_waiting_operations = node_embedding_num_waiting_operations.unsqueeze(1).expand([self.batch_size, n_node_features_machine, len_num_waiting_operations])

                node_embedding_machine_obs = self.node_representation(node_feature_machine)

                edge_index_machine = torch.tensor(edge_index_machine, dtype=torch.long, device=device)

                node_representation = self.func_machine_comm(node_embedding_machine_obs, edge_index_machine, n_node_features_machine,
                                                          mini_batch=mini_batch)
                node_representation = torch.cat([node_embedding_num_waiting_operations, node_representation], dim=2)

            """
            node_representation 
            - training 시        : batch_size X num_nodes X feature_size 
            - action sampling 시 : num_nodes X feature_size
            """
        if self.GNN == 'FastGTN':
            if mini_batch == False:
                with torch.no_grad():
                    node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)
                    A = self.get_heterogeneous_adjacency_matrix(edge_index_enemy, edge_index_ally)
                    node_representation = self.func_meta_path(A, node_feature, num_nodes = self.num_nodes, mini_batch = mini_batch)

            else:
                node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
                A = [self.get_heterogeneous_adjacency_matrix(edge_index_enemy[m], edge_index_ally[m]) for m in range(self.batch_size)]
                node_representation = self.func_meta_path(A, node_feature, num_nodes=self.num_nodes, mini_batch = mini_batch)
                #node_representation = torch.stack(node_representation, dim = 0).to(device)
        return node_representation

    def get_heterogeneous_adjacency_matrix(self, edge_index_enemy, edge_index_ally):
        A = []
        edge_index_enemy_transpose = deepcopy(edge_index_enemy)
        edge_index_enemy_transpose[1] = edge_index_enemy[0]
        edge_index_enemy_transpose[0] = edge_index_enemy[1]
        edge_index_ally_transpose = deepcopy(edge_index_ally)
        edge_index_ally_transpose[1] = edge_index_ally[0]
        edge_index_ally_transpose[0] = edge_index_ally[1]
        edges = [edge_index_enemy,
                 edge_index_enemy_transpose,
                 edge_index_ally,
                 edge_index_ally_transpose]
        for i, edge in enumerate(edges):
            edge = torch.tensor(edge, dtype = torch.long, device = device)
            value = torch.ones(edge.shape[1], dtype = torch.float, device = device)
            

            deg_inv_sqrt, deg_row, deg_col = _norm(edge.detach(),
                                                   self.num_nodes,
                                                   value.detach())  # row의 의미는 차원이 1이상인 node들의 index를 의미함

            value = deg_inv_sqrt[deg_row] * value  # degree_matrix의 inverse 중에서 row에 해당되는(즉, node의 차원이 1이상인) node들만 골라서 value_tmp를 곱한다
            A.append((edge, value))

        edge = torch.stack((torch.arange(0, self.num_nodes), torch.arange(0, self.num_nodes))).type(torch.cuda.LongTensor)
        value = torch.ones(self.num_nodes).type(torch.cuda.FloatTensor)
        A.append((edge, value))
        return A





    def cal_Q(self, obs, actions, avail_actions_next, agent_id, target = False):
        """

        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size

        """
        if target == False:
            obs_n = obs[:, agent_id]
            q = self.Q(obs_n)
            actions = torch.tensor(actions, device = device).long()
            act_n = actions[:, agent_id].unsqueeze(1)                    # action.shape : (batch_size, 1)
            q = torch.gather(q, 1, act_n).squeeze(1)                     # q.shape :      (batch_size, 1)
            return q
        else:
            with torch.no_grad():
                obs_next = obs
                obs_next = obs_next[:, agent_id]
                q_tar = self.Q_tar(obs_next)                        # q.shape :      (batch_size, action_size, 1)
                avail_actions_next = torch.tensor(avail_actions_next, device = device).bool()
                mask = avail_actions_next[:, agent_id]
                q_tar = q_tar.masked_fill(mask == 0, float('-inf'))
                q_tar_max = torch.max(q_tar, dim = 1)[0]
                return q_tar_max

    @torch.no_grad()
    def sample_action(self, node_representation, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """

        mask = torch.tensor(avail_action, device=device).bool()
        action = []
        for n in range(self.num_agent):
            obs = node_representation[n]
            Q = self.Q(obs)
            Q = Q.masked_fill(mask[n, :]==0, float('-inf'))
            greedy_u = torch.argmax(Q)
            mask_n = np.array(avail_action[n], dtype=np.float64)

            if np.random.uniform(0, 1) >= epsilon:
                u = greedy_u
                action.append(u)
            else:
                u = np.random.choice(self.action_space, p=mask_n / np.sum(mask_n))
                action.append(u)
        #print("sss", action)
        return action


    def learn(self, regularizer):
        node_features_machine, num_waiting_operations, edge_indices_machine, actions, rewards, dones, node_features_machine_next, num_waiting_operations_next, edge_indices_machine_next, avail_actions_next = self.buffer.sample()

        dummy = [0] * self.feature_size_job

        # max_job_length = np.max([np.max([len(nfj) for nfj in node_features_job]), np.max([len(nfj) for nfj in node_features_job_next])])
        # import time
        # start = time.time()
        # for nfj in node_features_job:
        #     if len(nfj) <= max_job_length:
        #         for j in range(max_job_length- len(nfj)):
        #             nfj.append(dummy)
        #
        # for nfj in node_features_job_next:
        #     if len(nfj) <= max_job_length:
        #         for j in range(max_job_length- len(nfj)):
        #             nfj.append(dummy)

        #remain_duration = time.time() - start



        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """

        #n_node_features_job = torch.tensor(node_features_job).shape[1]
        n_node_features_machine = torch.tensor(node_features_machine).shape[1]

        #start = time.time()
        obs = self.get_node_representation(
                                           node_features_machine,
                                            num_waiting_operations,
                                           edge_indices_machine,
                                           n_node_features_machine,
                                           mini_batch=True)

        obs_next = self.get_node_representation(
                                                node_features_machine_next,
                                                num_waiting_operations_next,
                                                edge_indices_machine_next,
                                                n_node_features_machine,
                                                mini_batch=True)
        #embedding_duration = time.time()-start
        dones = torch.tensor(dones, device = device, dtype = torch.float)
        rewards = torch.tensor(rewards, device = device, dtype = torch.float)
        import time
        start = time.time()
        q = [self.cal_Q(obs=obs,
                         actions=actions,
                         avail_actions_next=None,
                         agent_id=agent_id,
                         target=False) for agent_id in range(self.num_agent)]

        q_tar = [self.cal_Q(obs=obs_next,
                             actions=None,
                             avail_actions_next=avail_actions_next,
                             agent_id=agent_id,
                             target=True) for agent_id in range(self.num_agent)]

        q_duration = time.time() - start


        # start = time.time()
        q_tot = torch.stack(q, dim=1)
        q_tot_tar = torch.stack(q_tar, dim=1)
        q_tot = self.VDN(q_tot)
        q_tot_tar = self.VDN_target(q_tot_tar)
        td_target = rewards*self.num_agent + self.gamma* (1-dones)*q_tot_tar
        loss1 = F.huber_loss(q_tot, td_target.detach())
        loss = loss1
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, 10)
        self.optimizer.step()
        # if episode % 20 == 0 and episode > 0:
        #     self.Q_tar.load_state_dict(self.Q.state_dict())
        #     self.VDN_target.load_state_dict(self.VDN.state_dict())
        tau = 1e-3
        #backward_duration = time.time() - start

        #print("소요시간", remain_duration, embedding_duration, q_duration, backward_duration)

        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.VDN_target.parameters(), self.VDN.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

        return loss

