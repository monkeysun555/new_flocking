import numpy as np
import math
import matplotlib.pyplot as plt
import os 

N_USERS = 48
OPTIMIZE = 0
new_palette = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
                  '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
patterns = [ "/" , "|" , "\\"  , "-" , "+" , "x", "o", "O", ".", "*" ]

def process_user_info_line(line):
    elements = line.split(' ')
    user_id = int(elements[1])
    fov_trace = int(elements[4])
    bw_trace = float(elements[7])
    latency_group = int(elements[-1])
    return user_id, fov_trace, bw_trace, latency_group

def process_download_line(line):
    elements = line.split(' ')
    n_segs = int(elements[3])
    ave_rate = float(elements[6])
    freeze = float(elements[9])
    entropy = float(elements[12])
    ratio = float(elements[-1])
    return n_segs, ave_rate, freeze, entropy, ratio

def read_data_file(coor):
    if coor == 0:
        file_path = '../data/version_9/Mapping_usrs31_coor0_latency0.txt'       # Actually version 8 is 0/0 using LSTM
    elif coor == 1:
        file_path = '../data/version_10/Mapping_usrs31_coor1_latency0.txt'
    elif coor == 2:
        file_path = '../data/version_8/Mapping_usrs31_coor0_latency0.txt'
    elif coor == 3:
        file_path = '../data/version_11/Mapping_usrs31_coor1_latency0.txt'
    # Current only return entropy, this function can also plot average rate and freeze
    user_latency_groups = dict()
    entropys = []
    with open(file_path, 'r') as fr:
        cnt = 0
        curr_u_id = None
        for line in fr:
            if cnt%2 == 0:
                # User info line
                user_id, fov_trace, bw_trace, latency_group = process_user_info_line(line)
                user_latency_groups[user_id] = latency_group
                curr_u_id = user_id
                curr_l_id = latency_group
            else:
                n_segs, ave_rate, freeze, entropy, ratio = process_download_line(line)
                # entropys.append([curr_u_id, entropy, curr_l_id])
                entropys.append([curr_u_id, ratio, curr_l_id])
            cnt += 1
    entropys.sort(key = lambda x: (x[2]))
    return entropys

def plot_entropy(t1, t2, t3, t4):
    index = [str(user[0]) for user in t1]
    co0_data = [user[1] for user in t1]
    co1_data = [user[1] for user in t2]
    lstm = [user[1] for user in t3]
    lstm_c = [user[1] for user in t4]

    y_axis_upper = max(co0_data)*1.05
    n_group = 4
    group_len = 8
    co0_groups = []
    co1_groups = []
    # for g_id in range(n_group):
    #     co0_groups.append(co0_data[g_id*group_len:(g_id+1)*group_len])
    #     co1_groups.append(co1_data[g_id*group_len:(g_id+1)*group_len])
    
    # Position
    position = []
    curr_pos = 0
    for g_id in range(n_group):
        if g_id == 3:
            group_len = 7
        for u_id in range(group_len):
            position.append(curr_pos)
            curr_pos += 2.45
        curr_pos += 1.2
    group_len = 8
    width = 0.3  # the width of the bars
    pos_gap = 0.2
    opacity = 0.75

    p = plt.figure(figsize=(25,5))


    plt.bar(position, co0_data, color='none', width=width, edgecolor=new_palette[0], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP Self Prediction')
    plt.bar(position, co0_data, color='none', width=width, edgecolor='k', linewidth=0.5)

    plt.bar([x + width + pos_gap for x in position], co1_data, color='none', width=width, edgecolor=new_palette[1], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='TLP Collaborative')
    plt.bar([x + width + pos_gap for x in position], co1_data, color='none', width=width, edgecolor='k', linewidth=0.5)  


    plt.bar([x + 2*(width + pos_gap) for x in position], lstm, color='none', width=width, edgecolor=new_palette[2], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM Self Prediction')
    plt.bar([x + 2*(width + pos_gap)  for x in position], lstm, color='none', width=width, edgecolor='k', linewidth=0.5)    

    plt.bar([x + 3*(width + pos_gap) for x in position], lstm_c, color='none', width=width, edgecolor=new_palette[3], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM Collaborative Prediction')
    plt.bar([x + 3*(width + pos_gap)  for x in position], lstm_c, color='none', width=width, edgecolor='k', linewidth=0.5)   

    # rects0 = plt.bar(position, co0_data, width, alpha=opacity, color='b', label='co0')
    # rects1 = plt.bar([x + pos_gap for x in position], co1_data, width, alpha=opacity, color='g', label='co1')

    plt.legend(loc='upper right', fontsize=20, ncol=4)
    xticks_pos = [ position[group_id*group_len+4] - pos_gap for group_id in range(n_group)]
    plt.xticks(xticks_pos, ['Latency Group 1', 'Latency Group 2', 'Latency Group 3', 'Latency Group 4'], fontsize=20)
    plt.yticks(np.arange(0, y_axis_upper, 0.2), fontsize=20)
    plt.axis([-width, position[-1] + 4*(pos_gap+width), 0.2, 0.82])
    plt.ylabel('Tile Overlap Ratio', fontsize=20)
    plt.close()
    plt.tight_layout()
    return p

def main():
    coor = 0
    co0_entropys = read_data_file(coor)

    coor = 1
    co1_entropys = read_data_file(coor)

    coor = 2
    lstm = read_data_file(coor)

    coor = 3
    lstm_c = read_data_file(coor)
    # plot entropys for 48 users
    print(len(co0_entropys), len(co1_entropys))
    print(co0_entropys)
    print(co1_entropys)
    for i in range(len(co0_entropys)):
        assert co0_entropys[i][0] == co1_entropys[i][0]

    p = plot_entropy(co0_entropys, co1_entropys, lstm, lstm_c)
    p.savefig('../plots/kl_div.eps', format='eps', dpi=1000, figsize=(25, 5))

if __name__ == '__main__':
    main()