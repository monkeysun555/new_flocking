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

def get_entropy_pass(t):
    if t == 'h':
        path = RES_PATH + 'version_2/entropy/'
        gaps_paths = RES_PATH + 'version_2/entropy/'
    else:
        path = RES_PATH + 'version_7/entropy/'
        gaps_paths = RES_PATH + 'version_2/entropy/'

    return path

def collect_prediction_accuracy():
    accuracy_path = './fov_prediction/number_nss.txt'
    plot_datas = []
    count = 0
    with open(accuracy_path, 'r') as fr:
        for line in fr:
            line = line.split(' ')[:-1]
            # p_type = count//3
            # n_type = count%3
            # print(line)
            plot_datas.append([float(x) for x in line])
    print(plot_datas)
    return plot_datas

def collect_origin():
    accuracy_path = './fov_prediction/nss.txt'
    plot_datas = []
    count = 0
    with open(accuracy_path, 'r') as fr:
        for line in fr:
            line = line.split(' ')[:-1]
            # p_type = count//3
            # n_type = count%3
            # print(line)
            if count == 0 or count == 2:
                plot_datas.append([float(x) for x in line])
            count += 1
    print(plot_datas)
    return plot_datas

def plot_bar(types_data, odata):
    # print([x for x in types_data.keys])

    one_tlp = types_data[3]
    half_tlp = types_data[4]
    all_tlp = types_data[5]

    one_lstm = types_data[0]
    half_lstm = types_data[1]
    all_lstm = types_data[2]

    tlp_o = odata[1]
    lstm_o = odata[0]

    y_axis_upper = max(tlp_o)*1.05

    bar_width = 0.3
    gap = 0.05
    between = 0.1
    group_gap = 0.2

    tlp_o_p = []
    one_tlp_p = []
    half_tlp_p = []
    all_tlp_p = []

    lstm_o_p = []
    one_lstm_p = []
    half_lstm_p = []
    all_lstm_p = []

    x_label_pos = []
    curr_pos = 0.5*group_gap
    for i in range(5):
        curr_pos += 0.5*bar_width
        tlp_o_p += [curr_pos]
        curr_pos += bar_width + gap
        one_tlp_p += [curr_pos]
        curr_pos += bar_width + gap
        half_tlp_p += [curr_pos]
        curr_pos += bar_width + gap
        all_tlp_p += [curr_pos]       
        curr_pos += 0.5*(bar_width + between)
        x_label_pos += [curr_pos]
        curr_pos += 0.5*(bar_width + between)
        lstm_o_p += [curr_pos]
        curr_pos += bar_width + gap
        one_lstm_p += [curr_pos]
        curr_pos += bar_width + gap
        half_lstm_p += [curr_pos]
        curr_pos += bar_width + gap
        all_lstm_p += [curr_pos]       
        curr_pos += 0.5*bar_width + group_gap
    # x_positions = [1, 2, 3, 4, 5]
    
    p, ax = plt.subplots(1, 1, sharex=True, figsize=(12,8))


    # plot the same data on both axes
    # ax.plot(pts)
    # ax2.plot(pts)

    # zoom-in / limit the view to different portions of the data
    # ax.set_ylim(8., 20.)  # outliers only
    # ax2.set_ylim(0, 4)  # most of the data

    # hide the spines between ax and ax2
    # ax.spines['bottom'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax.xaxis.tick_top()
    # ax.tick_params(labeltop=False)  # don't put tick labels at the top
    # ax2.xaxis.tick_bottom()


    # ax2.bar(tlp_o_p, tlp_o, color='none', width=bar_width, edgecolor=new_palette[0], \
    #             hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP')
    # ax2.bar(tlp_o_p, tlp_o, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    # ax2.bar(one_tlp_p, one_tlp, color='none', width=bar_width, edgecolor=new_palette[1], \
    #             hatch=patterns[1]*6, linewidth=1.0, zorder = 0, label='Co-TLP (6)')
    # ax2.bar(one_tlp_p, one_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    # ax2.bar(half_tlp_p, half_tlp, color='none', width=bar_width, edgecolor=new_palette[2], \
    #             hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='Co-TLP (24)')
    # ax2.bar(half_tlp_p, half_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    # ax2.bar(all_tlp_p, all_tlp, color='none', width=bar_width, edgecolor=new_palette[3], \
    #             hatch=patterns[3]*6, linewidth=1.0, zorder = 0, label='Co-TLP (47)')
    # ax2.bar(all_tlp_p, all_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)


    # ax2.bar(lstm_o_p, lstm_o, color='none', width=bar_width, edgecolor=new_palette[4], \
    #             hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='LSTM$_{t}$')
    # ax2.bar(lstm_o_p, lstm_o, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    # ax2.bar(one_lstm_p, one_lstm, color='none', width=bar_width, edgecolor=new_palette[5], \
    #             hatch=patterns[1]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (6)')
    # ax2.bar(one_lstm_p, one_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)   

    # ax2.bar(half_lstm_p, half_lstm, color='none', width=bar_width, edgecolor=new_palette[6], \
    #             hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (24)')
    # ax2.bar(half_lstm_p, half_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    # ax2.bar(all_lstm_p, all_lstm, color='none', width=bar_width, edgecolor=new_palette[7], \
    #             hatch=patterns[3]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (47)')
    # ax2.bar(all_lstm_p, all_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)


    ax.bar(tlp_o_p, tlp_o, color='none', width=bar_width, edgecolor=new_palette[0], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='TLP')
    ax.bar(tlp_o_p, tlp_o, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    ax.bar(one_tlp_p, one_tlp, color='none', width=bar_width, edgecolor=new_palette[2], \
                hatch=patterns[1]*6, linewidth=1.0, zorder = 0, label='Co-TLP (6)')
    ax.bar(one_tlp_p, one_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    ax.bar(half_tlp_p, half_tlp, color='none', width=bar_width, edgecolor=new_palette[3], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='Co-TLP (18)')
    ax.bar(half_tlp_p, half_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    ax.bar(all_tlp_p, all_tlp, color='none', width=bar_width, edgecolor=new_palette[1], \
                hatch=patterns[3]*6, linewidth=1.0, zorder = 0, label='Co-TLP (30)')
    ax.bar(all_tlp_p, all_tlp, color='none', width=bar_width, edgecolor='k', linewidth=0.5)


    ax.bar(lstm_o_p, lstm_o, color='none', width=bar_width, edgecolor=new_palette[4], \
                hatch=patterns[0]*6, linewidth=1.0, zorder = 0, label='LSTM$_{s}$')
    ax.bar(lstm_o_p, lstm_o, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    ax.bar(one_lstm_p, one_lstm, color='none', width=bar_width, edgecolor=new_palette[6], \
                hatch=patterns[1]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (6)')
    ax.bar(one_lstm_p, one_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)   

    ax.bar(half_lstm_p, half_lstm, color='none', width=bar_width, edgecolor=new_palette[7], \
                hatch=patterns[2]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (18)')
    ax.bar(half_lstm_p, half_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)

    ax.bar(all_lstm_p, all_lstm, color='none', width=bar_width, edgecolor=new_palette[5], \
                hatch=patterns[3]*6, linewidth=1.0, zorder = 0, label='LSTM$_{c}$ (30)')
    ax.bar(all_lstm_p, all_lstm, color='none', width=bar_width, edgecolor='k', linewidth=0.5)


    ax.legend(loc='upper right', fontsize=20, ncol=2, labelspacing = -0.08, columnspacing = 0.7, framealpha = 0.)
    # xticks_pos = [0.5*(position[group_id*group_len+5]+position[group_id*group_len+6]) for group_id in range(n_group)]
    ax.set_xticks(x_label_pos)
    ax.set_yticks(np.arange(0.3, 1, 0.1))
    ax.set_xticklabels( ['1', '2', '3', '4', '5'], fontsize = 24)
    ax.set_yticklabels([x/100 for x in np.arange(30, 100, 10)], fontsize = 24)
    # ax.set_yticks(np.arange(8, 20, 4))
    # ax.set_yticklabels(np.arange(8, 20, 4), fontsize = 24)

    # plt.xticks(x_label_pos, ['1', '2', '3', '4', '5'], fontsize=22)
    # plt.yticks(np.arange(0, 12, 2), fontsize=22)
    # plt.axis([0, curr_pos, 0, 7.5])
    # plt.ylabel('KL Divergence', fontsize=24)
    # plt.xlabel('Prediction Interval (s)', fontsize=24)

    plt.axis([0, curr_pos, 0.3, 0.85])

    p.text(0.5, 0.025, 'Prediction Horizon (s)', ha='center', fontsize=24)
    p.text(0.04, 0.5, 'Tile Overlap Ratio', va='center', rotation='vertical', fontsize=24)

    # d = .015  # how big to make the diagonal lines in axes coordinates
    # # arguments to pass to plot, just so we don't keep repeating them
    # kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    # ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    # ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    # kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    # ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    # ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


    plt.close()
    return p
    


def main():
    types_data = collect_prediction_accuracy()
    origin_dagta = collect_origin()
    p = plot_bar(types_data, origin_dagta)
    p.show()
    input()
    p.savefig('./figures/fov_prediction/num_others_nss.eps', format='eps', dpi=1000, figsize=(30, 5))
    return
if __name__ == '__main__':
    main()