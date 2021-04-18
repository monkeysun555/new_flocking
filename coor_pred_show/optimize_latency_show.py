import numpy as np
import math
import matplotlib.pyplot as plt
import os 

N_USERS = 48
n_l_groups = 4
ADVANCE = 0

def process_user_info_line(line):
    elements = line.split(' ')
    user_id = int(elements[1])
    fov_trace = int(elements[4])
    bw_trace = int(elements[8])
    latency_group = int(elements[-1])
    return user_id, fov_trace, bw_trace, latency_group

def process_download_line(line):
    elements = line.split(' ')
    n_segs = int(elements[3])
    ave_rate = float(elements[6])
    freeze = float(elements[9])
    entropy = float(elements[12])
    return n_segs, ave_rate, freeze, entropy

def read_data_file_enhanced():

    # Coor0, 
    coor0_entropy, coor0_freeze, coor0_rate, \
    coor0_entropy_mig, coor0_freeze_mig, coor0_rate_mig = read_data_file(0, 0)
    print(coor0_entropy)
    # coor1/w.o.opt
    coor1_entropy, coor1_freeze, coor1_rate, \
    coor1_entropy_mig, coor1_freeze_mig, coor1_rate_mig  = read_data_file(1, 0)

    # coor1/withopt
    coor1opt_entropy, coor1opt_freeze, coor1opt_rate,\
    coor1opt_entropy_mig, coor1opt_freeze_mig, coor1opt_rate_mig = read_data_file(1, 1)

    print("Entropy")
    print("Coor0,               coor 1,         coor1optimized")
    for i in range(n_l_groups):
        print("group %s" %(str(i+1)))
        print('Freeze: ', np.round(np.mean(coor0_freeze[i]),2), np.round(np.mean(coor1_freeze[i]),2), np.round(np.mean(coor1opt_freeze[i]),2))
        print('Rate: ', np.round(np.mean(coor0_rate[i]),2), np.round(np.mean(coor1_rate[i]),2), np.round(np.mean(coor1opt_rate[i]),2))
        print('Entropy: ', np.round(np.mean(coor0_entropy[i]),2), np.round(np.mean(coor1_entropy[i]),2), np.round(np.mean(coor1opt_entropy[i]),2))
        if ADVANCE:
            len_1 = 0
            len_2 = 0
            len_3 = 0
            f_1 = 0
            f_2 = 0
            f_3 = 0
            if i in coor0_freeze_mig:
                len_1 = len(coor0_freeze_mig[i])
                f_1 = np.mean(coor0_freeze_mig[i])

            if i in coor1_freeze_mig:
                len_2 = len(coor1_freeze_mig[i])
                f_2 = np.mean(coor1_freeze_mig[i])

            if i in coor1opt_freeze_mig:
                len_3 = len(coor1opt_freeze_mig[i])
                f_3 = np.mean(coor1opt_freeze_mig[i])

            print('Num: ', np.round(len_1,2), np.round(len_2,2), np.round(len_3,2))
            print('# Freeze: ', np.round(f_1,2), np.round(f_2,2), np.round(f_3,2))
        print("<====================================>")

    # for i in range(n_l_groups):
    # GET download size
    download_path = '../data/user_' + str(N_USERS) + '/size/'
    #Get coor 0 
    coor0_user_size = []
    for u_id in range(N_USERS):
        coo0_path = download_path + 'user' + str(u_id) + '_coor0_latency0.txt'
        size = []
        with open(coo0_path, 'r') as r_file:
            for line in r_file:
                size.append(float(line.strip('\n').split(' ')[1]))
        coor0_user_size.append(np.mean(size))
    # coor 1
    coor1_user_size = []
    for u_id in range(N_USERS):
        coo1_path = download_path + 'user' + str(u_id) + '_coor1_latency0.txt'
        size = []
        with open(coo1_path, 'r') as r_file:
            for line in r_file:
                size.append(float(line.strip('\n').split(' ')[1]))
        coor1_user_size.append(np.mean(size))

    # coor1opt
    coor1opt_user_size = []
    for u_id in range(N_USERS):
        coo1opt_path = download_path + 'user' + str(u_id) + '_coor1_latency1.txt'
        size = []
        with open(coo1opt_path, 'r') as r_file:
            for line in r_file:
                size.append(float(line.strip('\n').split(' ')[1]))
        coor1opt_user_size.append(np.mean(size))

    # Show  overall average:
    print("For coor 0:")
    print("Average Size: ", np.round(np.mean(coor0_user_size),2))
    print("Average Rate: ", np.round(np.sum([np.sum(coor0_rate[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average Freeze: ", np.round(np.sum([np.sum(coor0_freeze[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average KL: ", np.round(np.sum([np.sum(coor0_entropy[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("<====================================>")
    print("For coor 1:")
    print("Average Size: ", np.round(np.mean(coor1_user_size),2))
    print("Average Rate: ", np.round(np.sum([np.sum(coor1_rate[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average Freeze: ", np.round(np.sum([np.sum(coor1_freeze[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average KL: ", np.round(np.sum([np.sum(coor1_entropy[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("<====================================>")
    print("For coor 1opt:")
    print("Average Size: ", np.round(np.mean(coor1opt_user_size),2))
    print("Average Rate: ", np.round(np.sum([np.sum(coor1opt_rate[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average Freeze: ", np.round(np.sum([np.sum(coor1opt_freeze[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("Average KL: ", np.round(np.sum([np.sum(coor1opt_entropy[l]) for l in range(n_l_groups)])/N_USERS,2))
    print("<====================================>")

def read_data_file(coor, optimize): 
    file_path = '../data/user_' + str(N_USERS) + '/' + 'Mapping_usrs' + str(N_USERS) + '_coor' + str(coor) + '_latency' + str(optimize) + '.txt'
    # Current only return entropy, this function can also plot average rate and freeze
    group_latency_dict = dict()
    group_freeze_dict = dict()
    group_rate_dict = dict()
    group_latency_mig = dict()
    group_freeze_mig = dict()
    group_rate_mig = dict()
    with open(file_path, 'r') as fr:
        cnt = 0
        curr_u_id = None
        for line in fr:
            if cnt%2 == 0:
                # User info line
                user_id, fov_trace, bw_trace, latency_group = process_user_info_line(line)
                curr_u_id = user_id
                curr_l_id = latency_group
            else:
                n_segs, ave_rate, freeze, entropy = process_download_line(line)
                # Change group if freeze is long
                new_latency_group = latency_group_change(freeze, curr_l_id)
                if ADVANCE:
                    if new_latency_group - curr_l_id >= 1:
                        if curr_l_id in group_latency_mig:
                            group_latency_mig[curr_l_id].append(entropy)
                            group_freeze_mig[curr_l_id].append(freeze)
                            group_rate_mig[curr_l_id].append(ave_rate)
                        else:
                            group_latency_mig[curr_l_id] = [entropy]
                            group_freeze_mig[curr_l_id] = [freeze]
                            group_rate_mig[curr_l_id] = [ave_rate]
                    else:
                        if new_latency_group in group_latency_dict:
                            group_latency_dict[new_latency_group].append(entropy)
                            group_freeze_dict[new_latency_group].append(freeze)
                            group_rate_dict[new_latency_group].append(ave_rate)
                        else:
                            group_latency_dict[new_latency_group] = [entropy]
                            group_freeze_dict[new_latency_group] = [freeze]
                            group_rate_dict[new_latency_group] = [ave_rate]
                else:
                    if curr_l_id in group_latency_dict:
                        group_latency_dict[curr_l_id].append(entropy)
                        group_freeze_dict[curr_l_id].append(freeze)
                        group_rate_dict[curr_l_id].append(ave_rate)
                    else:
                        group_latency_dict[curr_l_id] = [entropy]
                        group_freeze_dict[curr_l_id] = [freeze]
                        group_rate_dict[curr_l_id] = [ave_rate]
            cnt += 1
    
    return group_latency_dict, group_freeze_dict, group_rate_dict, group_latency_mig, group_freeze_mig, group_rate_mig

def latency_group_change(freeze, ori_latency_group):
    latency_gap = get_latency_gap(ori_latency_group)

    while freeze > latency_gap + 2.0:
        ori_latency_group += 1
        freeze -= latency_gap
        latency_gap = get_latency_gap(ori_latency_group)
    return ori_latency_group

def get_latency_gap(latency_group):
    if latency_group == 0:
        return 5.0
    elif latency_group == 1:
        return 6.0
    elif latency_group == 2:
        return 7.0
    elif latency_group == 3:
        return float('inf')
    else:
        assert 0 == 1

def main():
    read_data_file_enhanced()

if __name__ == '__main__':
    main()
