import pandas as pd
import numpy as np

# #시나리오 1
# num_job_type = 7
# num_machines = 19
# num_jobs = 70


# #시나리오 2
# num_job_type = 10
# num_machines = 29
# num_jobs = 100

# # #시나리오 3
num_job_type = 12
num_machines = 35
#num_jobs = 120

# num_job_type = 3
# num_machines = 4
# num_jobs = 10



def process_time_generator(lower, upper):
    return np.random.uniform(lower, upper)


def random_alter_machine_generator(num_machines):
    machine_list = [i for i in range(num_machines)]
    np.random.shuffle(machine_list)
    num = np.random.randint(int((num_machines-2)/2), int((num_machines+4)/2))
    result = machine_list[:num]
    return result

da = [0, 2, 4, 6, 7, 9, 11, 13, 15, 17, 18, 20, 22, 24, 25, 27, 29, 31, 33, 35, 36, 38, 40, 42, 44, 46, 48, 50, 51, 53, 55, 57, 58,60, 62]
wb = [i for i in range(63) if i not in da]
def setup_generator():
    setup_list = np.zeros([len(ops_name_list), len(ops_name_list)])
    for i in range(len(ops_name_list)):
        for j in range(len(ops_name_list)):
            if i == j:
                setup_list[i][j] = 0
            else:
                if j%2 == 0:
                    setup_list[i][j] = 1
                if j%2 == 1:
                    setup_list[i][j] = 3
        #
        # [[51.8 if i in
        #                 else 15.9 for j in range(len(ops_name_list))]
        #           for i in range(len(ops_name_list))]
    for i in range(len(ops_name_list)):
        setup_list[i][i] = 0
    return setup_list


ops_name_list = list()
process_time_list = list()

alternative_machine_list = list()
temp = [2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 8, 8]
for j in range(num_job_type):
    process_j_operation_list = list()
    machine_j_operation_list = list()
    num_operations = temp[j]
    for k in range(num_operations):
        ops_name_list.append("{}_{}".format(j, k))
        if k % 2 == 0:
            process_j_operation_list.append(1.6)
            machine_j_operation_list.append([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
        else:
            process_j_operation_list.append(4.8)
            machine_j_operation_list.append([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34])

    process_time_list.append(process_j_operation_list)
    alternative_machine_list.append(machine_j_operation_list)

if __name__ == "__main__":
    setup_list = setup_generator()

    df_name = pd.DataFrame(ops_name_list)
    df_process = pd.DataFrame(process_time_list)
    df_setup = pd.DataFrame(setup_list)
    df_machine = pd.DataFrame(alternative_machine_list)

    df_name.to_csv('dataset3_ops_name.csv', index=False)
    df_process.to_csv('dataset3_process_time.csv', index=False)
    df_setup.to_csv('dataset3_setup_time.csv', index=False)
    df_machine.to_csv('dataset3_alternative_machine.csv', index=False)