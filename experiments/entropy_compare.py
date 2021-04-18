import numpy as np
import math
import matplotlib.pyplot as plt
import os 

SHOW = 0
n_l_groups = 4
n_enhanced_version = [0, 2, 7]
label = ['MMSys', 'MMSys Decay', 'LSTM']
colors = ['r','orange','g']
n_users = 48

def process_user_info_line(line):
    elements = line.split(' ')
    user_id = int(elements[1])
    fov_trace = int(elements[4])
    bw_trace = float(elements[8])
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

    g_entropy, g_freeze, g_rate, u_entropy, u_rate, u_effective, u_l_group, max_start_idx, min_video_len, user_fovs, user_psnr = read_data_file()
    print(max_start_idx, min_video_len)
    compare_path = './compare/'
    if not os.path.isdir(compare_path):
        os.makedirs(compare_path)
    # Plot for each user
    for u in range(n_users):
        l_group = int(u_l_group[u])
        curr_path = compare_path + 'lg_' + str(l_group) + '_user_' + str(u) 
        figure = plt.figure(figsize=(10, 5))
        #  Entropy
        ave = [np.round(np.mean([x[1] for x in u_entropy[u][i] if x[0] >= max_start_idx and x[0] < min_video_len]), 2) for i in range(len(n_enhanced_version))]
        
        fov = user_fovs[u]
        t = [x[0] for x in fov if x[0] >= max_start_idx and x[0] < min_video_len]
        y = [x[1] for x in fov if x[0] >= max_start_idx and x[0] < min_video_len]
        p = [x[2] for x in fov if x[0] >= max_start_idx and x[0] < min_video_len]
        axy = plt.subplot(511)
        plt.plot(t,y,'.')
        axy = plt.subplot(512)
        plt.plot(t,p,'.')

        ax1 = plt.subplot(513)
        for i in range(len(n_enhanced_version)):
            plt.plot([x[0] for x in u_entropy[u][i]  if x[0] >= max_start_idx and x[0] < min_video_len ], \
                [x[1] for x in u_entropy[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ], color=colors[i], label = label[i] + ' (' + str(ave[i])+ ', 0)')
        
        plt.title('User ' + str(u) + ' in Latency Group: ' + str(l_group))
        # plt.legend(ncol=3)
        # if l_group == 0:
        #     # ax1.set_xlim([np.a,xmax])
        #     ymax = 20
        # elif l_group == 1:
        #     ymax = 5
        # elif l_group == 2:
        #     ymax = 4
        # elif l_group == 3:
        #     ymax = 3
        ymax=np.amax([x[1] for x in u_entropy[u][0]][10:100])*1.3

        ax1.set_ylim([0,ymax])
        # Rates/Effecitve

        ax3 = plt.subplot(514)
        eff = [np.round(np.mean([x[1] for x in u_rate[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ]), 2) for i in range(len(n_enhanced_version))]
        
        for i in range(len(n_enhanced_version)):
            plt.plot([x[0] for x in u_rate[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ], \
                [x[1] for x in u_rate[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ],color=colors[i],  label = label[i] + ' (' + str(eff[i]) + ', 0)')
        
        ax2 = plt.subplot(515)

        qualities = [np.round(np.mean([x[1] for x in user_psnr[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ]), 2) for i in range(len(n_enhanced_version))]
        
        for i in range(len(n_enhanced_version)):
            plt.plot([x[0] for x in user_psnr[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ], \
                [x[1] for x in user_psnr[u][i] if x[0] >= max_start_idx and x[0] < min_video_len ], color=colors[i],  label = label[i] +  ' (' + str(qualities[i]) + ', 0)')
        
        plt.legend(ncol=3)

        # ax2.set_xlim([xmin,xmax])
        ax2.set_ylim([0,np.amax([x[1] for x in user_psnr[u][-1]])*1.2])

        if SHOW:
            figure.show()
            input()
        figure.savefig(curr_path + '.eps', format='eps', dpi=1000, figsize=(10, 5))
        plt.close()





def read_data_file(): 
    entropy_dicts = {}
    freeze_dicts = {}
    rate_dicts = {}
    latency_dicts = {}

    for v in n_enhanced_version:
        file_path = '../data/version_' + str(v) + '/'
        mapping_path = file_path + 'Mapping_usrs48_coor1_latency1.txt'
        # Current only return entropy, this function can also plot average rate and freeze
        group_entropy_dict = dict()
        group_freeze_dict = dict()
        group_rate_dict = dict()
        group_latency_dict = dict()
        with open(mapping_path, 'r') as fr:
            cnt = 0
            curr_u_id = None
            for line in fr:
                if cnt%2 == 0:
                    # User info line
                    user_id, fov_trace, bw_trace, latency_group = process_user_info_line(line)
                    curr_u_id = user_id
                    curr_l_id = latency_group
                    group_latency_dict[user_id] = latency_group
                else:
                    n_segs, ave_rate, freeze, entropy = process_download_line(line)
                    # Change group if freeze is long
                    # new_latency_group = latency_group_change(freeze, curr_l_id)
                    # Add user to the group
                    if curr_l_id in group_entropy_dict:
                        group_entropy_dict[curr_l_id].append(entropy)
                        group_freeze_dict[curr_l_id].append(freeze)
                        group_rate_dict[curr_l_id].append(ave_rate)
                    else:
                        group_entropy_dict[curr_l_id] = [entropy]
                        group_freeze_dict[curr_l_id] = [freeze]
                        group_rate_dict[curr_l_id] = [ave_rate]

                cnt += 1
        entropy_dicts[v] = group_entropy_dict
        freeze_dicts[v] = group_freeze_dict
        rate_dicts[v] = group_rate_dict
        latency_dicts[v] = group_latency_dict


    # Read detailed entropy for each user
    users_entropy = []
    users_rate = []
    users_effective = []
    users_l_group = []
    user_l_quality = []
    max_start_idx, min_video_len = 0, 1000
    user_fovs = {}


    for user in range(n_users):
        curr_user_entropy = []
        curr_user_rate = []
        curr_user_effective = []
        curr_user_psnr = []
        # Check latency group for different methods are same
        l_group = None
        for en_v in n_enhanced_version:
            curr_v_entropy = []
            curr_v_rate = []
            curr_v_effective = []
            psnr = []
            if l_group == None:
                l_group = latency_dicts[en_v][user]
            # print(l_group, latency_dicts[en_v][user])
            assert latency_dicts[en_v][user] == l_group
            # Read detailed entropy info
            entropy_file = '../data/version_' + str(en_v) + '/entropy/user' + str(user) + '_coor1_latency1.txt'
            rate_file = '../data/version_' + str(en_v) + '/size/user' + str(user) + '_coor1_latency1.txt'
            effective_file = '../data/version_' + str(en_v) + '/rate/user' + str(user) + '_coor1_latency1.txt'
            quality = '../data/version_' + str(en_v) + '/wspsnr/user' + str(user) + '_coor1_latency1.txt'
            with open(entropy_file, 'r') as e_f:
                for line in e_f:
                    curr_v_entropy.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            with open(rate_file, 'r') as e_f:
                for line in e_f:
                    curr_v_rate.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            with open(effective_file, 'r') as e_f:
                for line in e_f:
                    curr_v_effective.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            with open(quality, 'r') as e_f:
                for line in e_f:
                    psnr.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            max_start_idx = max(max_start_idx, curr_v_entropy[0][0])
            min_video_len = min(min_video_len, curr_v_entropy[-1][0])
            print(min_video_len, user, l_group, en_v)
            curr_user_entropy.append(curr_v_entropy)
            curr_user_rate.append(curr_v_rate)
            curr_user_effective.append(curr_v_effective)
            curr_user_psnr.append(psnr)
            ## get fov
            fov = []
            if en_v == 7:
                fov_path = '../data/version_' + str(en_v) + '/fov/user' + str(user) + '_coor1_latency1.txt'
                with open(fov_path, 'r') as f_f:
                    for line in f_f:
                        line = line.strip('\n').split('\t')
                        t1, n1, n2, n3 = line[0].split(' ')
                        t2, n4, n5, n6 = line[1].split(' ')
                        fov += [[float(t1), float(n1), float(n2), float(n3)]]
                        fov += [[float(t2), float(n4), float(n5), float(n6)]]
                user_fovs[user] = fov
        users_entropy.append(curr_user_entropy)
        users_rate.append(curr_user_rate)
        users_effective.append(curr_user_effective)
        users_l_group.append(l_group)
        user_l_quality.append(curr_user_psnr)
    max_start_idx = max_start_idx + 5
    min_video_len = min_video_len - 5
    return entropy_dicts, freeze_dicts, rate_dicts, users_entropy, users_rate, users_effective, users_l_group, max_start_idx, min_video_len, user_fovs, user_l_quality

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
