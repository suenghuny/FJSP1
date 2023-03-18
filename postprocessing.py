import pandas as pd
import numpy as np

# iqn_result = pd.read_csv('(AL6) IQN/test_result_시나리오1.csv').loc[:, '0'].tolist()
# ppo_result = pd.read_csv('(AL3) PPO/test_result_시나리오1.csv').loc[:, '0'].tolist()
# dqn_result = pd.read_csv('(AL4) DQN/test_result_시나리오1.csv').loc[:, '0'].tolist()
# spsu_result = pd.read_csv('test_result_시나리오1.csv').loc[:, '0'].tolist()
# ssu_result = pd.read_csv('test_result_ssu_시나리오1.csv').loc[:, '0'].tolist()
#
# als = [iqn_result, ppo_result, dqn_result, spsu_result, ssu_result]
# cases = [1, 2, 3, 4, 5, 6, 7, 8]
#
# total_result = [[],[],[],[],[]]
# columns = ['IQN', 'PPO', 'DQN', 'RULE 1', 'RULE 2']
#
# for idx in range(5):
#     al = als[idx]
#     case8 = np.mean(al[0:299])
#     case7 = np.mean(al[300:599])
#     case6 = np.mean(al[600:899])
#     case5 = np.mean(al[900:1199])
#     case4 = np.mean(al[1200:1499])
#     case3 = np.mean(al[1500:1799])
#     case2 = np.mean(al[1800:2099])
#     case1 = np.mean(al[2100:2399])
#     total_result[idx].append(case1)
#     total_result[idx].append(case2)
#     total_result[idx].append(case3)
#     total_result[idx].append(case4)
#     total_result[idx].append(case5)
#     total_result[idx].append(case6)
#     total_result[idx].append(case7)
#     total_result[idx].append(case8)
#
# df = pd.DataFrame(total_result).T
# df.columns = columns
# df.index = cases
#
# print(df)
# df.to_csv('total_result_scenario_1_M18.csv')

iqn_result = pd.read_csv('(AL6) IQN/test_result_시나리오2.csv').loc[:, '0'].tolist()
ppo_result = pd.read_csv('(AL3) PPO/test_result_시나리오2.csv').loc[:, '0'].tolist()
dqn_result = pd.read_csv('(AL4) DQN/test_result_시나리오2.csv').loc[:, '0'].tolist()
spsu_result = pd.read_csv('test_result_시나리오2.csv').loc[:, '0'].tolist()
ssu_result = pd.read_csv('test_result_ssu_시나리오2.csv').loc[:, '0'].tolist()

als = [iqn_result, ppo_result, dqn_result, spsu_result, ssu_result]
cases = [1, 2, 3, 4, 5, 6, 7, 8]

total_result = [[],[],[],[],[]]
columns = ['IQN', 'PPO', 'DQN', 'RULE 1', 'RULE 2']

for idx in range(5):
    al = als[idx]
    case8 = np.mean(al[0:299])
    case7 = np.mean(al[300:599])
    case6 = np.mean(al[600:899])
    case5 = np.mean(al[900:1199])
    case4 = np.mean(al[1200:1499])
    case3 = np.mean(al[1500:1799])
    case2 = np.mean(al[1800:2099])
    case1 = np.mean(al[2100:2399])
    total_result[idx].append(case1)
    total_result[idx].append(case2)
    total_result[idx].append(case3)
    total_result[idx].append(case4)
    total_result[idx].append(case5)
    total_result[idx].append(case6)
    total_result[idx].append(case7)
    total_result[idx].append(case8)

df = pd.DataFrame(total_result).T
df.columns = columns
df.index = cases

print(df)
df.to_csv('total_result_scenario_2_M18.csv')