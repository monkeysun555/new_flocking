import numpy as np
import math
import matplotlib.pyplot as plt
import os 

NUM_USER = 48


new_palette = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
                  '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
patterns = [ "/" , "|" , "\\"  , "-" , "+" , "x", "o", "O", ".", "*" ]

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

def get_user_freezing(coor, optimize):
    mapping_path = '../data/user_' + str(NUM_USER) + '/Mapping_usrs' + str(NUM_USER) + '_coor' + str(coor) + '_latency' + str(optimize) + '.txt'

    kl_dict = []
    freeze_dict = []
    with open(mapping_path, 'r') as fr:
        cnt = 0
        curr_u_id = None
        for line in fr:
            if cnt%2 == 0:
                # User info line
                user_id, fov_trace, bw_trace, latency_group = process_user_info_line(line)
                curr_u_id = user_id
            else:
                n_segs, ave_rate, freeze, entropy = process_download_line(line)
                kl_dict.append(entropy)
                freeze_dict.append(freeze)
            cnt += 1
    return kl_dict, freeze_dict

def plot_freeze_cdf(freeze_0, freeze_1, freeze_2):
    assert len(freeze_0) == len(freeze_1)
    assert len(freeze_0) == len(freeze_2)
    max_freeze = max(max(freeze_0), max(freeze_1), max(freeze_2))
    min_freeze = min(min(freeze_0), min(freeze_1), min(freeze_2))
    latency_interval = 0.2

    curr_lat = min_freeze
    new_freeze0 = []
    new_freeze1 = []
    new_freeze2 = []
    x_pos = []
    while curr_lat <= max_freeze:
        new_freeze0.append(np.sum([1.0 for x in freeze_0 if x <= curr_lat])/len(freeze_0))
        new_freeze1.append(np.sum([1.0 for x in freeze_1 if x <= curr_lat])/len(freeze_1))
        new_freeze2.append(np.sum([1.0 for x in freeze_2 if x <= curr_lat])/len(freeze_2))
        x_pos.append(curr_lat)
        curr_lat += latency_interval

    p = plt.figure(figsize=(15, 10))
    plt.plot(x_pos, new_freeze0, color=new_palette[0], label="Single User Prediction", linewidth=2, alpha=0.9)
    plt.plot(x_pos, new_freeze1, color=new_palette[1], label="Collaborative Prediction", linewidth=2, alpha=0.9)
    plt.plot(x_pos, new_freeze2, color=new_palette[2], label="Flocking Strategy", linewidth=2, alpha=0.9)
   
    plt.legend(loc='upper right', fontsize=24, ncol=2)
    plt.xticks(np.arange(int(min_freeze/2)*2, max_freeze, 2), fontsize=24)
    plt.yticks(np.arange(0, 1, 0.2), fontsize=24)
    plt.xlabel('Freeze (s)', fontsize=24)
    plt.ylabel('CDF', fontsize=24)
    plt.axis([min_freeze, max_freeze, 0, 1.2])
    plt.close()

    return p

def plot_entropy_cdf(kl_0, kl_1, kl_2):
    assert len(kl_0) == len(kl_1)
    assert len(kl_0) == len(kl_2)
    max_kl = max(max(kl_0), max(kl_1), max(kl_2))
    min_kl = min(min(kl_0), min(kl_1), min(kl_2))
    kl_interval = 0.2

    curr_kl = min_kl
    new_kl_0 = []
    new_kl_1 = []
    new_kl_2 = []
    x_pos = []
    while curr_kl <= max_kl:
        new_kl_0.append(np.sum([1.0 for x in kl_0 if x <= curr_kl])/len(kl_0))
        new_kl_1.append(np.sum([1.0 for x in kl_1 if x <= curr_kl])/len(kl_1))
        new_kl_2.append(np.sum([1.0 for x in kl_2 if x <= curr_kl])/len(kl_2))
        x_pos.append(curr_kl)
        curr_kl += kl_interval

    p = plt.figure(figsize=(15, 10))
    plt.plot(x_pos, new_kl_0, color=new_palette[0], label="Single User Prediction", linewidth=2, alpha=0.9)
    plt.plot(x_pos, new_kl_1, color=new_palette[1], label="Collaborative Prediction", linewidth=2, alpha=0.9)
    plt.plot(x_pos, new_kl_2, color=new_palette[2], label="Flocking Strategy", linewidth=2, alpha=0.9)
   
    plt.legend(loc='upper right', fontsize=24, ncol=2)
    plt.xticks(np.arange(int(min_kl/2)*2, max_kl, 2), fontsize=24)
    plt.yticks(np.arange(0, 1, 0.2), fontsize=24)
    plt.xlabel('KL Divergence', fontsize=24)
    plt.ylabel('CDF', fontsize=24)
    plt.axis([min_kl, max_kl, 0, 1.2])
    plt.close()

    return p

def main():
    cdf_path = '../plots/user_' + str(NUM_USER) + '/'
    if not os.path.isdir(cdf_path):
        os.makedirs(cdf_path)

    # Plot freezing cdf
    coor = 0
    optimize = 0 
    kl_0, freeze_0 = get_user_freezing(coor, optimize)

    coor = 1
    optimize = 0
    kl_1, freeze_1 = get_user_freezing(coor, optimize)

    coor = 1
    optimize = 1
    kl_2, freeze_2 = get_user_freezing(coor, optimize)

    p = plot_freeze_cdf(freeze_0, freeze_1, freeze_2)
    p.savefig(cdf_path + 'freeze_cdf.eps', format='eps', dpi=1000, figsize=(15, 10))

    p = plot_entropy_cdf(kl_0, kl_1, kl_2)
    p.savefig(cdf_path + 'entropy_cdf.eps', format='eps', dpi=1000, figsize=(15, 10))



if __name__ == '__main__':
    main()