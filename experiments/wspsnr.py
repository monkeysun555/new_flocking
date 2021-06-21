import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os 

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

SHOW = 0
n_l_groups = 4
n_enhanced_version = [0, 8, 11, 12, 7]
# label = ['Heuristic', 'D-Heuristic', 'LSTM$_{t/c}^{-a}$', 'LSTM$_{t/c}^{-d}$', 'LSTM$_{t/c}$']
label = ['H-Flocking', 'Self-Pre', 'Co-Pre', 'Prop', 'Flocking']

# n_enhanced_version = [0, 2, -2, 7]
# label = ['Heuristic', 'D-Heuristic', 'LSTM$_{t/c}^{-}$', 'LSTM$_{t/c}$']
colors = ['r','#17becf', 'purple', 'g', 'orange']
n_users = 31

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
    return n_segs, ave_rate, freeze, entropy

def process_bold(matrix, find_max=True):
    new_str_matrix = []
    for i in range(len(matrix)):
        new_row = []
        if find_max:
            optimal = max(matrix[i])
        else:
            optimal = min(matrix[i])

        for v in matrix[i]:
            if v != optimal:
                new_row += [str(v)]
            else:
                new_row += ['{\\bf ' + str(v) + '}']
        new_str_matrix += [new_row]
    return new_str_matrix


def plot_cdf(users_info, plot_type, path):
    cdf_max = max([max(v) for v in users_info.values()])
    cdf_min = min([min(v) for v in users_info.values()])

    gap = (cdf_max - cdf_min)/200
    cdf_x = np.arange(int(cdf_min), int(np.ceil(cdf_max)), gap)

    tick_count = 5
    tick_gap = (cdf_max - cdf_min)/tick_count

    p = plt.figure(figsize=(7,5.5))
    for i in range(len(n_enhanced_version)):
        v = n_enhanced_version[i]
        info = users_info[v]
        name = label[i]

        curr_cdf = []
        for point in cdf_x:
            curr_cdf.append(len([x for x in info if x <= point])/float(len(info)))
        plt.plot(cdf_x, curr_cdf, color = colors[i], label = name)

    plt.legend(loc='best',fontsize = 26, ncol=1, frameon=False, labelspacing=0.)
    plt.xlabel(plot_type, fontweight='bold', fontsize=26)
    if plot_type == 'Quality':
        plt.xticks(range(int(cdf_min), int(np.ceil(cdf_max))+2, 5), range(int(cdf_min), int(np.ceil(cdf_max))+2, 5), fontsize=22)
    else:
        plt.xticks(fontsize=22)
    plt.yticks(np.arange(0, 1.001, 0.2), fontsize=22)
    plt.ylabel('CDF', fontweight='bold', fontsize=22)
    
    # plt.axis([int(qoe_lower/X_GAP)*X_GAP, int(qoe_upper/X_GAP)*X_GAP+ X_GAP, 0.001, 1])
    p.set_tight_layout(True)
    p.savefig(path + plot_type.split(' ')[0] + '.eps', format='eps', dpi=1000, figsize=(7, 5.5))

def read_data_file_enhanced():
    g_entropy, g_freeze, g_rate, u_entropy, u_rate, u_effective, \
    u_l_group, max_start_idx, min_video_len, user_fovs, user_psnr, \
    groups_psnr, groups_ratios, cdf_freezes, cdf_wspsnr, group_bws, group_nss = read_data_file()


    ### Show TMM bw
    for l in range(4):
        print(np.mean(group_bws[l]), np.std(group_bws[l]))

    # Get data first
    kls = [[] for _ in range(5)]
    freezes = [[] for _ in range(5)]
    psnrs = [[] for _ in range(5)]
    ratios = [[] for _ in range(5)]
    nss = [[] for _ in range(5)]

    for i in range(len(n_enhanced_version)):
        v_name = n_enhanced_version[i]
        for j in range(4):
            # 4 latnecy groups
            kls[j] += [np.round(np.mean(g_entropy[v_name][j]), 2)]
            freezes[j] += [np.round(np.mean(g_freeze[v_name][j]) ,2)]
            psnrs[j] += [np.round(np.mean(groups_psnr[v_name][j]), 1)]
            ratios[j] += [np.round(np.mean(groups_ratios[v_name][j]), 2)]
            nss[j] += [np.round(np.mean(group_nss[v_name][j]), 2)]

    # print(kls, freezes, psnrs)

    for i in range(len(n_enhanced_version)):
        kls[-1] += [np.round(np.mean([kls[j][i] for j in range(4)]),2)]
        freezes[-1] += [np.round(np.mean([freezes[j][i] for j in range(4)]),2)]
        psnrs[-1] += [np.round(np.mean([psnrs[j][i] for j in range(4)]),1)]
        ratios[-1] += [np.round(np.mean([ratios[j][i] for j in range(4)]),2)]
        nss[-1] += [np.round(np.mean([nss[j][i] for j in range(4)]),2)]


    str_kl = process_bold(kls, False)
    str_freeze = process_bold(freezes, False)
    str_qua = process_bold(psnrs)
    str_ra = process_bold(ratios)
    str_nss = process_bold(nss)

    # 
    latex = []
    for i in range(len(n_enhanced_version)):
        curr_line = '{ \\bf ' +  label[i] + '} & '
        for l_g in range(5):
            # curr_line += ' ' + str_kl[l_g][i] + ' & '
            curr_line += ' ' + str_nss[l_g][i] + ' & '
            # curr_line += ' ' + str_ra[l_g][i] + ' & '
            curr_line += ' ' + str_freeze[l_g][i] + ' & '
            curr_line += ' ' + str_qua[l_g][i] + ' &'
        latex += [curr_line[:-1] + ' \\\\']
    for line in latex:
        print(line)

    print(ratios)


    ## PLOT CDF of different metrics (quality/freeze)
    cdf_path = './cdf/'
    if not os.path.isdir(cdf_path):
        os.makedirs(cdf_path)
    plot_cdf(cdf_freezes, 'Freeze (s)', cdf_path)
    plot_cdf(cdf_wspsnr, 'Quality (db)', cdf_path)

    # return
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

    cdf_freezes = {}
    cdf_wspsnr = {}

    
    for v in n_enhanced_version:
        if v in [0, 2, 7, 12]:
            file_path = '../data/version_' + str(v) + '/'
            mapping_path = file_path + 'Mapping_usrs31_coor1_latency1.txt'
        elif v == -1:
            file_path = '../data/version_7_0latency/'
            mapping_path = file_path + 'Mapping_usrs31_coor1_latency0.txt'
        elif v == -2:
            file_path = '../data/version_7_0rate/'
            mapping_path = file_path + 'Mapping_usrs31_coor1_latency1.txt'
        elif v == 8:
            print('8')
            file_path = '../data/version_' + str(v) + '/'
            mapping_path = file_path + 'Mapping_usrs31_coor0_latency0.txt'
        elif v == 11:
            # print('8')
            file_path = '../data/version_' + str(v) + '/'
            mapping_path = file_path + 'Mapping_usrs31_coor1_latency0.txt'

        # Current only return entropy, this function can also plot average rate and freeze
        group_entropy_dict = {}
        group_freeze_dict = {}
        group_rate_dict = {}
        group_latency_dict = {}
        curr_version_freeze = []
        group_bw_dict = {}
        # group_psnr_dict = {}
        print(mapping_path)
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

                    curr_version_freeze += [freeze]
                    # Change group if freeze is long
                    # new_latency_group = latency_group_change(freeze, curr_l_id)
                    # Add user to the group
                    if curr_l_id in group_entropy_dict:
                        group_entropy_dict[curr_l_id].append(entropy)
                        group_freeze_dict[curr_l_id].append(freeze)
                        group_rate_dict[curr_l_id].append(ave_rate)
                        group_bw_dict[curr_l_id].append(bw_trace)
                    else:
                        group_entropy_dict[curr_l_id] = [entropy]
                        group_freeze_dict[curr_l_id] = [freeze]
                        group_rate_dict[curr_l_id] = [ave_rate]
                        group_bw_dict[curr_l_id] = [bw_trace]
                cnt += 1
        cdf_freezes[v] = curr_version_freeze
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
    max_start_idx, min_video_len = 15, 1000
    user_fovs = {}

    groups_psnr = {}
    groups_ratios = {}
    groups_nss = {}

    for user in range(n_users):
        curr_user_entropy = []
        curr_user_rate = []
        curr_user_effective = []
        curr_user_psnr = []
        curr_user_ratio = []
        # if user not in groups_psnr:
        #   groups_psnr[user] = {}
        # Check latency group for different methods are same
        l_group = None
        for en_v in n_enhanced_version:
            if en_v not in groups_psnr:
                groups_psnr[en_v] = {}

            if en_v not in groups_ratios:
                groups_ratios[en_v] = {}

            if en_v not in groups_nss:
                groups_nss[en_v] = {}

            if en_v not in cdf_wspsnr:
                cdf_wspsnr[en_v] = []

            curr_v_entropy = []
            curr_v_rate = []
            curr_v_effective = []
            psnr = []
            curr_ratio = []
            nss = []
            if l_group == None:
                l_group = latency_dicts[en_v][user]
            # print(l_group, latency_dicts[en_v][user])
            # assert latency_dicts[en_v][user] == l_group

            if l_group not in groups_psnr[en_v]:
                groups_psnr[en_v][l_group] = []

            if l_group not in groups_ratios[en_v]:
                groups_ratios[en_v][l_group] = []

            if l_group not in groups_nss[en_v]:
                groups_nss[en_v][l_group] = []

            # Read detailed entropy info
            if en_v in [0, 2, 7, 3, 12]:
                entropy_file = '../data/version_' + str(en_v) + '/entropy/user' + str(user) + '_coor1_latency1.txt'
                rate_file = '../data/version_' + str(en_v) + '/size/user' + str(user) + '_coor1_latency1.txt'
                effective_file = '../data/version_' + str(en_v) + '/rate/user' + str(user) + '_coor1_latency1.txt'
                quality = '../data/version_' + str(en_v) + '/wspsnr/user' + str(user) + '_coor1_latency1.txt'
                tile_ratio = '../data/version_' + str(en_v) + '/ratios/user' + str(user) + '_coor1_latency1.txt'
                nss_scores = '../data/version_' + str(en_v) + '/nss/user' + str(user) + '_coor1_latency1.txt'
            elif en_v == -1:
                entropy_file = '../data/version_7_0latency/entropy/user' + str(user) + '_coor1_latency0.txt'
                rate_file = '../data/version_7_0latency/size/user' + str(user) + '_coor1_latency0.txt'
                effective_file = '../data/version_7_0latency/rate/user' + str(user) + '_coor1_latency0.txt'
                quality = '../data/version_7_0latency/wspsnr/user' + str(user) + '_coor1_latency0.txt'
                tile_ratio = '../data/version_7_0latency/ratios/user' + str(user) + '_coor1_latency0.txt'
                nss_scores = '../data/version_7_0latency/nss/user' + str(user) + '_coor1_latency0.txt'

            elif en_v == -2:
                entropy_file = '../data/version_7_0rate/entropy/user' + str(user) + '_coor1_latency1.txt'
                rate_file = '../data/version_7_0rate/size/user' + str(user) + '_coor1_latency1.txt'
                effective_file = '../data/version_7_0rate/rate/user' + str(user) + '_coor1_latency1.txt'
                quality = '../data/version_7_0rate/wspsnr/user' + str(user) + '_coor1_latency1.txt'
                tile_ratio = '../data/version_7_0rate/ratios/user' + str(user) + '_coor1_latency1.txt'
                nss_scores = '../data/version_7_0rate/nss/user' + str(user) + '_coor1_latency1.txt'
            elif en_v == 8:
                entropy_file = '../data/version_' + str(en_v) + '/entropy/user' + str(user) + '_coor0_latency0.txt'
                rate_file = '../data/version_' + str(en_v) + '/size/user' + str(user) + '_coor0_latency0.txt'
                effective_file = '../data/version_' + str(en_v) + '/rate/user' + str(user) + '_coor0_latency0.txt'
                quality = '../data/version_' + str(en_v) + '/wspsnr/user' + str(user) + '_coor0_latency0.txt'
                tile_ratio = '../data/version_' + str(en_v) + '/ratios/user' + str(user) + '_coor0_latency0.txt'
                nss_scores = '../data/version_' + str(en_v) + '/nss/user' + str(user) + '_coor0_latency0.txt'
            elif en_v == 11:
                entropy_file = '../data/version_' + str(en_v) + '/entropy/user' + str(user) + '_coor1_latency0.txt'
                rate_file = '../data/version_' + str(en_v) + '/size/user' + str(user) + '_coor1_latency0.txt'
                effective_file = '../data/version_' + str(en_v) + '/rate/user' + str(user) + '_coor1_latency0.txt'
                quality = '../data/version_' + str(en_v) + '/wspsnr/user' + str(user) + '_coor1_latency0.txt'
                tile_ratio = '../data/version_' + str(en_v) + '/ratios/user' + str(user) + '_coor1_latency0.txt'
                nss_scores = '../data/version_' + str(en_v) + '/nss/user' + str(user) + '_coor1_latency0.txt'

            with open(entropy_file, 'r') as e_f:
                for line in e_f:
                    # curr_v_entropy.append((int(line.strip().split()[0]), float(line.strip().split()[1])))
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

            with open(tile_ratio, 'r') as e_f:
                for line in e_f:
                    curr_ratio.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            with open(nss_scores, 'r') as e_f:
                for line in e_f:
                    nss.append((int(line.strip().split()[0]), float(line.strip().split()[1])))

            # print(psnr)
            max_start_idx = max(max_start_idx, curr_v_entropy[0][0])
            min_video_len = min(min_video_len, curr_v_entropy[-1][0])
            print(min_video_len, user, l_group, en_v)
            # if l_group == 2:
            #     if en_v == 0:
            #         curr_v_entropy += offset_l3_hr[0]
            #     elif en_v == 2:
            #         curr_v_entropy += offset_l3_hr[1]
            curr_user_entropy.append(curr_v_entropy)
            curr_user_rate.append(curr_v_rate)
            curr_user_effective.append(curr_v_effective)
            curr_user_psnr.append(psnr)

            low, high = 20, 200
            wspsnr_data = [x[1] for x in psnr if low<x[0]<high]
            # base = high - low
            base = len(wspsnr_data)
            if en_v == -1:
                cdf_wspsnr[en_v] += [np.sum(wspsnr_data)/base]
                groups_psnr[en_v][l_group] += [np.sum(wspsnr_data)/base]
            else:
                groups_psnr[en_v][l_group] += [np.sum(wspsnr_data)/base]
                cdf_wspsnr[en_v] += [np.sum(wspsnr_data)/base]
            
            groups_ratios[en_v][l_group] += [np.mean([x[1] for x in curr_ratio if low<x[0]<high])]
            groups_nss[en_v][l_group] += [np.mean([x[1] for x in nss if low<x[0]<high])]
            
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
    return entropy_dicts, freeze_dicts, rate_dicts, users_entropy, users_rate, users_effective, users_l_group, \
            max_start_idx, min_video_len, user_fovs, user_l_quality, groups_psnr, groups_ratios, cdf_freezes, cdf_wspsnr, group_bw_dict, groups_nss

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
