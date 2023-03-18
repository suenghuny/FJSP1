import pandas as pd
import numpy as np

import math

soe = 0.01
problem = 3  #'시나리오2', '시나리오3'
mode = 'test3' #'training' #'test1', 'test2', 'test3'

if problem == 1:
    if mode == 'training':
        code = 1
    if mode == 'test1':
        code = 11
    if mode == 'test2':
        code = 12
    if mode == 'test3':
        code = 13

if problem == 2:
    if mode == 'training':
        code = 2
    if mode == 'test1':
        code = 21
    if mode == 'test2':
        code = 22
    if mode == 'test3':
        code = 23

if problem == 3:
    if mode == 'training':
        code = 3
    if mode == 'test1':
        code = 31
    if mode == 'test2':
        code = 32
    if mode == 'test3':
        code = 33


try:
    process_time_list = pd.read_csv('C:\\Users\\sueng\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_process_time.csv'.format(problem)).values.tolist()
    ops_name_array = pd.read_csv('C:\\Users\\sueng\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_ops_name.csv'.format(problem)).values
    ops_name_array = np.reshape(ops_name_array, (1, -1))
    ops_name_list = ops_name_array[0].tolist()

    alternative_machine_list = pd.read_csv('C:\\Users\\sueng\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_alternative_machine.csv'.format(code)).values.tolist()

    setup_list = pd.read_csv('C:\\Users\\sueng\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_setup_time.csv'.format(problem)).values.tolist()
except FileNotFoundError as FE:
    process_time_list = pd.read_csv(
        'C:\\Users\\User\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_process_time.csv'.format(problem)).values.tolist()
    ops_name_array = pd.read_csv(
        'C:\\Users\\User\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_ops_name.csv'.format(problem)).values
    ops_name_array = np.reshape(ops_name_array, (1, -1))
    ops_name_list = ops_name_array[0].tolist()
    alternative_machine_list = pd.read_csv(
        'C:\\Users\\User\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_alternative_machine.csv'.format(
            code)).values.tolist()

    setup_list = pd.read_csv(
        'C:\\Users\\User\\OneDrive - SNU\\FJSP_MARL\\scheduling_problems\\dataset{}_setup_time.csv'.format(problem)).values.tolist()

maxlen = len(process_time_list[0])
for k in range(len(process_time_list)):
    for t in range(maxlen):
        if math.isnan(process_time_list[k][-1]):
            del process_time_list[k][-1]

#print(len(alternative_machine_list[0]))
maxlen = len(alternative_machine_list[0])
for k in range(len(alternative_machine_list)):
    for t in range(maxlen):
        try:
            if math.isnan(alternative_machine_list[k][-1]):
                del alternative_machine_list[k][-1]
        except TypeError as TE:pass

#print(alternative_machine_list[0])
# print(process_time_list)
#print(np.array(setup_list).shape)
#print(setup_list)
#시나리오 1
# num_job_type = 7
# num_machines = 10
# num_jobs = 70
#

# #시나리오 2
# num_job_type = 10
# num_machines = 20
# num_jobs = 100
#
# #시나리오 3



if problem == 3:
    num_job_type = 12
    if mode == 'training':
        num_machines = 30
    if mode == 'test1':
        num_machines = 40
    if mode == 'test2':
        num_machines = 50
    if mode == 'test3':
        num_machines = 60
    num_jobs = 120
    sp1 = [10]*num_job_type
    sp2 = [10, 9, 11, 9, 11, 10, 11, 10, 10, 9, 10, 10]
    sp3 = [10, 9, 11, 9, 10, 11, 11, 10, 9, 10, 10, 10]
    sp3 = [9, 10, 10, 11, 11, 10, 11, 10, 10, 9, 10, 10]
    sp4 = [10, 9, 11, 9, 10, 11, 11, 10, 9, 10, 10, 10]
    sp5 = [11, 9, 10, 11, 9, 10, 9, 11, 11, 11, 9, 9]
    sp6 = [9, 9 ,9, 10, 10, 10, 11, 11, 11, 10, 9, 11]
    sp7 = [10, 11, 10, 9, 10, 11, 9, 9 ,11, 10, 9, 11]
    sp8 = [11, 9, 11, 9, 11, 11, 9, 9, 10, 10, 10, 10]
    sp9 = [10, 10, 11, 9, 11, 10, 10, 9, 9, 10, 11, 10]
    sp10 = [9, 11, 10, 11, 9, 10, 11, 9, 9, 10, 11, 10]
    sp11 = [9, 11, 10, 11, 9, 10, 11, 9, 11, 10, 10, 9]
    sp12 = [11, 10, 10, 10, 9, 9, 9, 11, 11, 11, 9, 10]
    sp13 = [11, 9, 9, 10, 11, 11, 9, 11, 10, 11, 9, 11]
    sp14 = [9, 11, 9, 11, 9, 11, 9, 11, 9, 11, 9, 11]
    sp15 = [10, 10, 9, 9, 11, 11, 9, 11, 10, 9, 10, 11]
    sp16 = [9, 10, 11, 9, 10, 9, 11, 11, 10, 9, 10, 11]
    sp17 = [10, 11, 9, 10, 11, 11, 9, 9, 10, 11, 9, 10]
    sp18 = [11, 9, 11, 11, 9, 9, 9, 11, 10, 9, 10, 11]
    sp19 = [10, 10, 11, 11, 10, 10, 10, 9, 10, 10, 10, 9]
    sp20 = [11, 9, 11, 10, 9, 11, 10, 9, 11, 10, 9, 10]
    problems = [sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10,
                sp11, sp12, sp13, sp14, sp15, sp16, sp17, sp18, sp19, sp20]
    # for i in problems:
    #     print(sum(i))
if problem == 2:
    num_job_type = 10
    if mode == 'training':
        num_machines = 20
    if mode == 'test1':
        num_machines = 23
    if mode == 'test2':
        num_machines = 26
    if mode == 'test3':
        num_machines = 29
    num_jobs = 100
    sp1 = [10]*num_job_type
    sp2 = [10, 9, 11, 9, 11, 10, 11, 10, 10, 9]
    sp3 = [10, 9, 11, 9, 10, 11, 11, 10, 9, 10]
    sp3 = [9, 10, 10, 11, 11, 10, 11, 10, 9, 9]
    sp4 = [10, 9, 11, 9, 10, 11, 11, 10, 9, 10]
    sp5 = [11, 9, 10, 11, 9, 10, 9, 11, 10, 10]
    sp6 = [9, 9 ,9, 10, 10, 10, 11, 11, 11, 10]
    sp7 = [10, 11, 10, 9, 10, 11, 9, 9 ,11, 10]
    sp8 = [11, 9, 11, 9, 11, 11, 9, 9, 10, 10]
    sp9 = [10, 10, 11, 9, 11, 10, 10, 9, 9, 11]
    sp10 = [9, 11, 10, 11, 9, 10, 11, 9, 9, 11]
    sp11 = [9, 11, 10, 11, 9, 10, 11, 9, 10, 10]
    sp12 = [11, 10, 10, 10, 9, 9, 9, 11, 11, 10]
    sp13 = [11, 9, 9, 10, 11, 11, 9, 11, 9, 10]
    sp14 = [9, 11, 9, 11, 9, 11, 9, 11, 9, 11]
    sp15 = [10, 10, 10, 10, 11, 11, 9, 11, 9, 9]
    sp16 = [9, 10, 11, 9, 10, 9, 11, 11, 10, 10]
    sp17 = [10, 11, 9, 10, 11, 11, 9, 9, 10, 10]
    sp18 = [11, 9, 11, 11, 9, 9, 9, 11, 10, 10]
    sp19 = [10, 10, 11, 11, 10, 9, 9, 9, 10, 11]
    sp20 = [11, 9, 11, 10, 9, 11, 10, 9, 11, 9]
    problems = [sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10,
                sp11, sp12, sp13, sp14, sp15, sp16, sp17, sp18, sp19, sp20]
if problem == 1:
    num_job_type = 7
    if mode == 'training':
        num_machines = 10
    if mode == 'test1':
        num_machines = 13
    if mode == 'test2':
        num_machines = 16
    if mode == 'test3':
        num_machines = 19
    num_jobs = 70
    sp1 = [10]*num_job_type
    sp2 = [10, 9, 11, 9, 11, 10, 10]
    sp3 = [10, 9, 11, 9, 10, 11, 10]
    sp3 = [9, 10, 10, 11, 11, 10, 9]
    sp4 = [10, 9, 11, 9, 10, 11, 10]
    sp5 = [11, 9, 10, 11, 9, 10, 10]
    sp6 = [9, 11 ,11, 10, 10, 9, 10]
    sp7 = [10, 11, 10, 9, 10, 11, 9]
    sp8 = [11, 9, 11, 9, 10, 10, 10]
    sp9 = [10, 9, 10, 9, 11, 10, 11]
    sp10 = [9, 11, 10, 11, 9, 10, 10]
    sp11 = [9, 11, 10, 11, 9, 10, 10]
    sp12 = [11, 10, 11, 11, 9, 9, 9]
    sp13 = [11, 9, 9, 10, 11, 11, 9]
    sp14 = [9, 11, 9, 11, 9, 11, 10]
    sp15 = [10, 10, 10, 10, 11, 10, 9]
    sp16 = [10, 10, 11, 10, 10, 9, 10]
    sp17 = [10, 11, 9, 10, 11, 10, 9]
    sp18 = [11, 10, 11, 11, 9, 9, 9]
    sp19 = [10, 10, 11, 11, 10, 9, 9]
    sp20 = [11, 9, 11, 10, 9, 11, 9]
    problems = [sp1, sp2, sp3, sp4, sp5, sp6, sp7, sp8, sp9, sp10,
                sp11, sp12, sp13, sp14, sp15, sp16, sp17, sp18, sp19, sp20]
