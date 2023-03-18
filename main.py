import pandas as pd

from GDN import Agent
from functools import partial
import numpy as np

import sys
import os
import time
from Environment_Multi.FJSP_multi import RL_ENV
from cfg import get_cfg
cfg = get_cfg()

vessl_on = cfg.vessl
if vessl_on == True:
    import vessl
    vessl.init()
else:
    from torch.utils.tensorboard import SummaryWriter






map_name1 = cfg.map_name
GNN = cfg.GNN
heterogenous = False

"""
Protoss
colossi : 200.0150.01.0
stalkers : 80.080.00.625
zealots : 100.050.00.5

Terran
medivacs  : 150.00.00.75
marauders : 125.00.00.5625
marines   : 45.00.00.375

Zerg
zergling : 35.00.00.375
hydralisk : 80.00.00.625
baneling : 30.00.00.375
spine crawler : 300.00.01.125`
"""




def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer):
    env.reset()
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False
    start = time.time()



    while not done:
        # self.get_node_feature_job(), self.get_node_feature_machine(), self.get_edge_index_job_machine(), self.get_edge_index_machine_machine()
        node_feature_machine,edge_index_machine = env.get_heterogeneous_graph()

        # print(np.array(node_feature_job).shape, np.array(node_feature_machine).shape)
        #  node_feature_job, node_feature_machine, edge_index_job, edge_index_machine, n_node_features, mini_batch

        n_node_feature_machine = np.array(node_feature_machine).shape[0]
        if GNN == 'GAT':
            node_representation = agent.get_node_representation(
                                                                node_feature_machine,
                                                                edge_index_machine,
                                                                n_node_feature_machine,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm

        avail_action = env.get_avail_actions()
        action = agent.sample_action(node_representation, avail_action, epsilon)
        reward, done, info = env.step(action)
        agent.buffer.memory(node_feature_machine,edge_index_machine, action, reward, done, avail_action)
        episode_reward += reward

        t += 1
        step += 1
        if (t % 5000 == 0) and (t >0):
            eval = True

        if e >= train_start:
            loss = agent.learn(regularizer=0)
            losses.append(loss.detach().item())
        if epsilon >= min_epsilon:
            epsilon = epsilon - anneal_epsilon
        else:
            epsilon = min_epsilon
        #print(episode_reward, done)
    if e >= train_start:
        print("Total reward in episode {} = {}, epsilon : {}, time_step : {}, episode_duration : {}".format(
                                                                                                e,
                                                                                                np.round(episode_reward, 3),
                                                                                                np.round(epsilon, 3),
                                                                                                t, np.round(time.time()-start, 3)))
    return episode_reward, epsilon, t, eval

def main():

    env1 = RL_ENV()
    #env1.reset()


    hidden_size_obs = cfg.hidden_size_obs       # GAT 해당(action 및 node representation의 hidden_size)
    hidden_size_comm = cfg.hidden_size_comm
    hidden_size_Q = cfg.hidden_size_Q         # GAT 해당
    hidden_size_meta_path = cfg.hidden_size_meta_path # GAT 해당
    n_representation_job = cfg.n_representation_job  # GAT 해당
    n_representation_machine = cfg.n_representation_machine
    buffer_size = cfg.buffer_size
    batch_size = cfg.batch_size
    gamma = cfg.gamma
    learning_rate = cfg.lr
    n_multi_head = cfg.n_multi_head
    dropout = cfg.dropout
    num_episode = cfg.num_episode
    train_start = cfg.train_start
    epsilon = cfg.epsilon
    min_epsilon = cfg.min_epsilon
    anneal_steps = cfg.anneal_steps
    teleport_probability = cfg.teleport_probability
    gtn_beta = cfg.gtn_beta
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps
    if vessl_on == True:

        output_dir = "output/"
    else:
        output_dir = "output/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_dir = 'output/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    initializer = True
    # writer = SummaryWriter(log_dir,
    #                        comment="map_name_{}_GNN_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}.csv".format(
    #                            map_name1, GNN, learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs,
    #                            n_representation_comm))

    # "n_agents": num_machines,
    # "obs_shape": 7 + num_jobs + max_ops_length,  # + self.n_agents,
    # "n_actions": len(ops_name_list) + 1

    agent1 = Agent(num_agent=env1.get_env_info()["n_agents"],
                   feature_size_job=env1.get_env_info()["job_feature_shape"],
                   feature_size_machine=env1.get_env_info()["machine_feature_shape"],
                   hidden_size_meta_path = hidden_size_meta_path,
                   hidden_size_obs=hidden_size_obs,
                   hidden_size_comm=hidden_size_comm,
                   hidden_size_Q=hidden_size_Q,
                   n_multi_head=n_multi_head,
                   n_representation_job=n_representation_job,
                   n_representation_machine=n_representation_machine,
                   dropout=dropout,
                   action_size=env1.get_env_info()["n_actions"],
                   buffer_size=buffer_size,
                   batch_size=batch_size,
                   learning_rate=learning_rate,
                   gamma=gamma,
                   GNN=GNN,
                   teleport_probability = teleport_probability,
                   gtn_beta = gtn_beta)


    t = 0
    epi_r = []
    win_rates = []
    for e in range(num_episode):
        episode_reward, epsilon, t, eval = train(agent1, env1, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer)
        initializer = False
        epi_r.append(episode_reward)
        #writer.add_scalar("episode_reward/train", episode_reward, e)
        if e % 200 <= 0.1:
            if vessl_on == True:
                agent1.save_model(output_dir+"{}.pt".format(t))
            else:
                agent1.save_model(output_dir+"{}.pt".format(t))
        if e % 100 == 1:
            if vessl_on == True:
                vessl.log(step = e, payload = {'reward' : np.mean(epi_r)})
                epi_r = []
                r_df= pd.DataFrame(epi_r)
                r_df.to_csv(output_dir+"reward.csv")
            else:
                r_df= pd.DataFrame(epi_r)
                r_df.to_csv(output_dir+"reward.csv")
        #
        # if eval == True:
        #     win_rate = evaluation(env1, agent1, 32)
        #     win_rates.append(win_rate)
        #     if vessl_on == True:
        #         vessl.log(step = t, payload = {'win_rate' : win_rate})
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")
        #     else:
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")






main()

