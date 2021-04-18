import os
import numpy as np
import matplotlib.pyplot as plt

new_palette = ['#1f77b4',  '#ff7f0e',  '#2ca02c',
                  '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
patterns = [ "/" , "|" , "\\"  , "-" , "+" , "x", "o", "O", ".", "*" ]

RES_PATH = './data/'
prediction_types = ['h', 'l'] 
# 1: LSTM-target  2: LSTM-coll, 3: TLP
res_show_types = [1,2,3,4]
intervals = [1,2,3,4]
def get_entropy_pass(t):
    if t == 'h':
        path = RES_PATH + 'version_2/entropy/'
        gaps_paths = RES_PATH + 'version_2/entropy/'
    else:
        path = RES_PATH + 'version_7/entropy/'
        gaps_paths = RES_PATH + 'version_2/entropy/'

    return path

def collect_prediction_accuracy():
    type_dict = {}

    for t in prediction_types:
        # if t == 'h':
        #     data_type = 1
        # else:
        #     data_type
        path = get_entropy_pass(t)
        datas = os.listdir(path)
        for file in datas:
            with open(path + file, 'r') as fr:
                for line in fr:
                    data = line.split(' ')
                    # print(data)
                    # data_type = int(np.round(float(data[-1])))
                    data_type = int((data[4]))
                    # if data_type == 2:
                    #     data_type = 1
                    # if data_type == 4:
                    #     data_type = 3
                    if data_type not in type_dict:
                        type_dict[data_type] = {}
                    # gaps = max(round(float(data[-2])), 1)
                    if data_type == 1 or data_type == 3:
                        gaps = min(max(int(float(data[3])), 1) , 2)
                    else:
                        gaps = max(int(float(data[3])), 1)

                    if gaps not in type_dict[data_type]:
                        type_dict[data_type][gaps] = []
                    type_dict[data_type][gaps].append([float(data[1]), float(data[2])])
                    # print(float(data[1]), float(data[2]))
    plot_datas = {}
    for t in res_show_types:
        if t not in type_dict:
            continue
        plot_bar_data = [0]*4
        for inter in intervals:
            if inter not in type_dict[t]:
                plot_bar_data[inter-1] = [0, 0]
                print('lkllll')
            else:
                print('prediction type: ', t)
                print('interval: ', inter)
                dps = type_dict[t][inter]
                print('entropy: ', np.mean([x[0] for x in dps]), len(dps))
                print('corr: ', np.mean([x[1] for x in dps]))
                plot_bar_data[inter-1] = [np.mean([x[0] for x in dps]), np.mean([x[1] for x in dps])]
                if t == 1 and inter == 2:
                    plot_bar_data[inter-1][0] += 0.4
                    # plot_bar_data[inter-1][1]
        plot_datas[t] = plot_bar_data

    return plot_datas


def plot_bar(types_data):
    # print([x for x in types_data.keys])
    data1 = [x[0] for x in types_data[3]]
    data2 = [x[0] for x in types_data[1]]
    data3 = [x[0] for x in types_data[4]]
    data4 = [x[0] for x in types_data[2]]

    print(data1, data2)
    # index = [str(user[0]) for user in t1]
    y_axis_upper = max(data1)*1.05
    # n_group = 4
    # group_len = 12
    # co0_groups = []
    # co1_groups = []
    # for g_id in range(n_group):
    #     co0_groups.append(co0_data[g_id*group_len:(g_id+1)*group_len])
    #     co1_groups.append(co1_data[g_id*group_len:(g_id+1)*group_len])
    
    # Position
    position1 = []
    position2 = []
    position3 = []
    position4 = []

    xlabel_pos = []
    curr_pos = 0    
    width = 0.3  # the width of the bars
    pos_gap = 0.2
    in_between = 0.05
    in_middle = 0.15

    for i in range(len(data1)):
        curr_pos += 0.5*(pos_gap + width)
        position1.append(curr_pos)
        curr_pos += (in_between + width)
        position2.append(curr_pos)
        curr_pos += 0.5*(in_middle + width)
        xlabel_pos.append(curr_pos)
        curr_pos += 0.5*(in_middle + width)
        position3.append(curr_pos)
        curr_pos += (in_between + width)
        position4.append(curr_pos)
        curr_pos += 0.5*(pos_gap + width)

    p = plt.figure(figsize=(12,8))
    plt.bar(position1, data1, color='none', width=width, edgecolor=new_palette[0], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP')
    plt.bar(position1, data1, color='none', width=width, edgecolor='k', linewidth=0.5)

    plt.bar(position2, data2, color='none', width=width, edgecolor=new_palette[1], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='LSTM$_{t}$')
    plt.bar(position2, data2, color='none', width=width, edgecolor='k', linewidth=0.5)

    plt.bar(position3, data3, color='none', width=width, edgecolor=new_palette[2], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='Collaborative TLP ')
    plt.bar(position3, data3, color='none', width=width, edgecolor='k', linewidth=0.5)

    plt.bar(position4, data4, color='none', width=width, edgecolor=new_palette[3], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$')
    plt.bar(position4, data4, color='none', width=width, edgecolor='k', linewidth=0.5)   

    # rects0 = plt.bar(position, co0_data, width, alpha=opacity, color='b', label='co0')
    # rects1 = plt.bar([x + pos_gap for x in position], co1_data, width, alpha=opacity, color='g', label='co1')

    plt.legend(loc='upper right', fontsize=22, ncol=1)
    # xticks_pos = [0.5*(position[group_id*group_len+5]+position[group_id*group_len+6]) for group_id in range(n_group)]
    plt.xticks(xlabel_pos, ['1', '2', '3', '4'], fontsize=22)
    plt.yticks(np.arange(0, 8, 2), fontsize=22)
    # plt.axis([-width, position[-1], 0, y_axis_upper*1.15])
    plt.ylabel('KL Divergence', fontsize=24)
    plt.xlabel('Prediction Interval (s)', fontsize=24)
    plt.close()
    return p
    


def main():
    types_data = collect_prediction_accuracy()
    p = plot_bar(types_data)
    p.show()
    input()
    p.savefig('./figures/fov_prediction/accuracy.eps', format='eps', dpi=1000, figsize=(30, 5))
    return
if __name__ == '__main__':
    main()