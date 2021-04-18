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
    accuracy_path = './fov_prediction/accuracy.txt'
    plot_datas = []
    with open(accuracy_path, 'r') as fr:
        for line in fr:
            line = line.split(' ')[:-1]
            # print(line)
            plot_datas.append([float(x) for x in line])
    print(plot_datas)
    return plot_datas


def plot_bar(types_data):
    # print([x for x in types_data.keys])
    data1 = types_data[2]
    data2 = types_data[3]
    data3 = types_data[0] # add 0.5
    data4 = types_data[1]

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
    in_middle = 0.05

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


    p, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,8))


    # plot the same data on both axes
    # ax.plot(pts)
    # ax2.plot(pts)

    # zoom-in / limit the view to different portions of the data
    ax.set_ylim(8., 16.)  # outliers only
    ax2.set_ylim(0, 4.)  # most of the data

    # hide the spines between ax and ax2
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()



    # p = plt.figure(figsize=(12,8))
    ax2.bar(position1, data1, color='none', width=width, edgecolor=new_palette[0], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP')
    ax2.bar(position1, data1, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax2.bar(position2, data2, color='none', width=width, edgecolor=new_palette[1], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='Co-TLP')
    ax2.bar(position2, data2, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax2.bar(position3, data3, color='none', width=width, edgecolor=new_palette[2], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{t}$')
    ax2.bar(position3, data3, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax2.bar(position4, data4, color='none', width=width, edgecolor=new_palette[3], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$')
    ax2.bar(position4, data4, color='none', width=width, edgecolor='k', linewidth=0.5) 


    ax.bar(position1, data1, color='none', width=width, edgecolor=new_palette[0], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP')
    ax.bar(position1, data1, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax.bar(position2, data2, color='none', width=width, edgecolor=new_palette[1], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='Co-TLP')
    ax.bar(position2, data2, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax.bar(position3, data3, color='none', width=width, edgecolor=new_palette[2], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{t}$')
    ax.bar(position3, data3, color='none', width=width, edgecolor='k', linewidth=0.5)

    ax.bar(position4, data4, color='none', width=width, edgecolor=new_palette[3], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$')
    ax.bar(position4, data4, color='none', width=width, edgecolor='k', linewidth=0.5)  

    # rects0 = plt.bar(position, co0_data, width, alpha=opacity, color='b', label='co0')
    # rects1 = plt.bar([x + pos_gap for x in position], co1_data, width, alpha=opacity, color='g', label='co1')

    print(xlabel_pos)
    ax.legend(loc='upper left', fontsize=22, ncol=2, framealpha = 0, labelspacing = -0.08, columnspacing = 0.7)
    # xticks_pos = [0.5*(position[group_id*group_len+5]+position[group_id*group_len+6]) for group_id in range(n_group)]
    ax2.set_xticks(xlabel_pos)
    ax2.set_yticks(np.arange(0, 5, 2))
    # ax2.axis([0, position4[-1]+width, 0, 7])
    ax2.set_xticklabels( ['1', '2', '3', '4', '5'], fontsize = 24)
    ax2.set_yticklabels(np.arange(0, 5, 2), fontsize = 24)
    ax.set_yticks(np.arange(8, 20, 4))
    ax.set_yticklabels(np.arange(8, 20, 4), fontsize = 24)

    # ax2 = plt.axes()
    plt.axis([0, position4[-1]+width, 0, 4])

    p.text(0.5, 0.025, 'Prediction Interval (s)', ha='center', fontsize=24)
    p.text(0.05, 0.5, 'KL Divergence ', va='center', rotation='vertical', fontsize=24)

    # plt.ylabel('', fontsize=24)
    # plt.xlabel('', fontsize=24)

    # ax = plt.axes()
    # ax.axis([0, position4[-1]+width, 8, 20])

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # ax2.close()
    return p
    


def main():
    types_data = collect_prediction_accuracy()
    p = plot_bar(types_data)
    p.show()
    input()
    p.savefig('./figures/fov_prediction/accuracy_new.eps', format='eps', dpi=1000, figsize=(30, 5))
    return
if __name__ == '__main__':
    main()