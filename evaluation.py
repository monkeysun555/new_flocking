import numpy as np
from config import Config
import matplotlib.pyplot as plt
import os 
import pickle as pk

TOTAL_USER = 31
PLOT_USER =  10
VID = 49
COOR = 1
OPTIMIZE = 1
TYPE = 3    # 3, -2, 7
if OPTIMIZE:
    UPPER_BOUND = 5.0               # 2, 3 ,3 ,3  and 2, 4, 5, 6
else:
    UPPER_BOUND = 2.0

LATS = [2, 6, 10, 15]

def plot_entropy(entropys):
    seg_offset = entropys[0][0]-1
    y_axis_upper = max([x[1] for x in entropys[3:-2]])
    y_plot_upper = 25
    # if y_axis_upper >= 20:
    #     gap = 5
    #     y_plot_upper = 25
    # elif y_axis_upper >= 10:
    #     gap = 3
    #     y_plot_upper = 15
    # elif y_axis_upper >= 5:
    #     gap = 1
    #     y_plot_upper = 10
    # else:
    #     y_plot_upper = 5
    #     gap = 0.5
    new_entropy_seg = [entropys[0][0] - seg_offset]
    new_entropy = [entropys[0][1]]
    for i in range(1, len(entropys)-2):
        new_entropy_seg.append(entropys[i][0]-seg_offset-1e-4)
        new_entropy.append(entropys[i-1][1])
        new_entropy_seg.append(entropys[i][0]-seg_offset)
        new_entropy.append(entropys[i][1])
    x_upper = new_entropy_seg[-1]

    p = plt.figure(figsize=(20,5))
    plt.plot(new_entropy_seg, new_entropy, color='chocolate', linewidth=2, alpha=0.9)
    
    # plt.legend(loc='upper right', fontsize=28, ncol=2)
    plt.xticks(np.arange(0, x_upper, 25), fontsize=28)
    plt.yticks(np.arange(0, 30.1, 10), fontsize=28)
    # plt.yticks(np.arange(0, 1.00001, 0.5), fontsize=28)
    plt.ylabel('KL Divergence', fontsize=24)
    plt.xlabel('Time (s)', fontsize=28)
    plt.axis([0, x_upper, 0, y_plot_upper])
    plt.tight_layout()
    plt.close()

    return p

def plot_ratio(entropys):
    seg_offset = entropys[0][0]-1
    y_axis_upper = max([x[1] for x in entropys[3:-2]])

    # if y_axis_upper >= 20:
    #     gap = 5
    # elif y_axis_upper >= 10:
    #     gap = 3
    # elif y_axis_upper >= 5:
    #     gap = 1
    # else:
    #     gap = 0.5
    new_entropy_seg = [entropys[0][0] - seg_offset]
    new_entropy = [entropys[0][1]]
    for i in range(1, len(entropys)-2):
        new_entropy_seg.append(entropys[i][0]-seg_offset-1e-4)
        new_entropy.append(entropys[i-1][1])
        new_entropy_seg.append(entropys[i][0]-seg_offset)
        new_entropy.append(entropys[i][1])
    x_upper = new_entropy_seg[-1]

    p = plt.figure(figsize=(20,5))
    plt.plot(new_entropy_seg, new_entropy, color='chocolate', label="KL Divergence", linewidth=2, alpha=0.9)
    
    plt.legend(loc='upper right', fontsize=28, ncol=2)
    plt.xticks(np.arange(0, x_upper, 25), fontsize=28)
    plt.yticks(np.arange(0, 30.1, 10), fontsize=28)
    # plt.ylabel('', fontsize=24)
    plt.xlabel('Time (s)', fontsize=28)
    plt.axis([0, x_upper, -1, 1])
    plt.tight_layout()
    plt.close()

    return p

def plot_bw_rates(bw_trace, rates, size_trace):
    bw_offset = bw_trace[0][0]-1
    seg_offset = rates[0][0]-1
    size_seg_offset = size_trace[0][0]-1

    x_upper = max(rates[-1][0]-seg_offset, size_trace[-1][0]-size_seg_offset)
    y_axis_upper = max([x[1] for x in bw_trace])
    new_rate_seg = [rates[0][0] - seg_offset]
    new_rate = [rates[0][1]]
    for i in range(1, len(rates)):
        new_rate_seg.append(rates[i][0]-seg_offset-1e-4)
        new_rate.append(rates[i-1][1])
        new_rate_seg.append(rates[i][0]-seg_offset)
        new_rate.append(rates[i][1])

    new_size_seg = [size_trace[0][0]-size_seg_offset]
    new_size = [size_trace[0][1]]
    for i in range(1, len(size_trace)-1):
        new_size_seg.append(size_trace[i][0]-size_seg_offset-1e-4)
        new_size.append(size_trace[i-1][1])
        new_size_seg.append(size_trace[i][0]-size_seg_offset)
        new_size.append(size_trace[i][1])



    p = plt.figure(figsize=(20,5))
    plt.plot([x[0] - bw_offset for x in bw_trace], [x[1] for x in bw_trace], linestyle='-', color='gray', label="Bandwidth", linewidth=2, alpha=0.1)
    # plt.plot(new_size_seg, new_size, color='green', label="Delivered Rates", linewidth=2, alpha=0.9)
    # plt.plot(new_rate_seg, new_rate, color='chocolate', label="Effective Rates", linewidth=2, alpha=0.9)
    
    plt.legend(loc='upper right', fontsize=28, ncol=3)
    plt.xticks(np.arange(0, x_upper, 25), fontsize=28)
    plt.yticks(np.arange(0, y_axis_upper, 200), fontsize=28)
    plt.xlabel('Time (s)', fontsize=28)
    plt.ylabel('Rate (Mbps)', fontsize=28)
    plt.axis([0, x_upper, 0, y_axis_upper*1.25])
    plt.tight_layout()
    plt.close()

    return p

def plot_buffer(buffers, freezes):
    y_axis_upper = max([x[1] for x in buffers])
    seg_offset = freezes[0][0]-1
    new_buffers_time = [buffers[0][0]]
    new_buffers = [buffers[0][1]]

    # There is no wait time included:
    added_time = 0.0
    for i in range(len(buffers)):
        buffers[i][0] += added_time
        if buffers[i][1] > UPPER_BOUND:
            added_time += buffers[i][1] - UPPER_BOUND
    x_upper = buffers[-2][0]
    freeze_upper = freezes[-1][0] - seg_offset
    time_factor = freeze_upper/x_upper

    # initial no buffer assumption
    for i in range(1,3):
        new_buffers_time.append(buffers[i][0]-1e-4)
        new_buffers.append(buffers[i-1][1])
        new_buffers_time.append(buffers[i][0])
        new_buffers.append(buffers[i][1])

    pre_buffer = new_buffers[-1]
    pre_time = new_buffers_time[-1]
    wait_time = 0.0
    for i in range(3, len(buffers)-1):
        time_gap = buffers[i][0] - pre_time
        new_buffers_time.append(buffers[i][0]-1e-4)
        new_buffers.append(max(pre_buffer-time_gap, 0.0))
        new_buffers_time.append(buffers[i][0])
        new_buffers.append(buffers[i][1])
        pre_buffer = new_buffers[-1]
        pre_time = new_buffers_time[-1]

    actual_freezes = [[x[0]- seg_offset-1, x[1]] for x in freezes if x[1] > 0]
    ## PLot alignment
    buffer_zero_points = [new_buffers_time[i]*time_factor+0.5 for i in range(len(new_buffers_time)) if new_buffers[i] == 0]
    print(actual_freezes)
    print(buffer_zero_points)

    p = plt.figure(figsize=(20,5))
    plt.plot([x*time_factor for x in  new_buffers_time], new_buffers, color='chocolate', label="Buffer Size", linewidth=2, alpha=0.8)
    
    # plt.bar([x[0]-seg_offset-1 for x in freezes], [x[1] for x in freezes], color='blue', label="Freeze")
    plt.bar(buffer_zero_points, [x[1] for x in actual_freezes], color='blue', label="Freeze")

    plt.legend(loc='upper right', fontsize=28, ncol=2)
    plt.xticks(np.arange(0, freeze_upper+1, 25), fontsize=28)
    plt.yticks(np.arange(0, y_axis_upper, 1), fontsize=28)
    plt.xlabel('Time(s)', fontsize=28)
    plt.ylabel('Buffer Length (s)', fontsize=28)
    plt.axis([0, freeze_upper, 0, y_axis_upper*1.3])
    plt.tight_layout()    
    plt.close()

    return p

def plot_fov(time, yaw, pitch):
    c = 0.8
    offset = time[0]
    x_upper = int(time[-1]) - offset
    p = plt.figure(figsize=(20,5))
    plt.plot([x - offset for x in time], yaw, linestyle='-', color='green', label="Yaw", linewidth=2, alpha=0.9)
    plt.plot([x - offset for x in time], [0.5*np.pi - c*(0.5*np.pi - x) for x in pitch], linestyle='-', color='chocolate', label="Pitch", linewidth=2, alpha=0.9)
    # plt.plot(new_rate_seg, new_rate, color='chocolate', label="Effective Rates", linewidth=2, alpha=0.9)
    
    plt.legend(loc='upper right', fontsize=28, ncol=3)
    plt.xticks(np.arange(0, x_upper, 25), fontsize=28)
    plt.yticks(np.arange(-np.pi, np.pi+0.001, 0.5*np.pi), [-180, -90, 0, 90, 180], fontsize=28)
    plt.xlabel('Time (s)', fontsize=28)
    plt.ylabel('FoV', fontsize=28)
    plt.axis([0, x_upper, -np.pi-0.2, np.pi+0.2])
    plt.tight_layout()
    plt.close()

    return p
def main():

    if TYPE == 7:
        data_path = './data/version_7/'
        system_info_path =  data_path + 'Mapping_usrs' + str(TOTAL_USER) +  '_coor1_latency1.txt'
        OPTIMIZE = 1
    elif TYPE == 3:
        data_path = './data/version_3/'
        system_info_path =  data_path + 'Mapping_usrs' + str(TOTAL_USER) +  '_coor1_latency1.txt'
        OPTIMIZE = 1
    elif TYPE == -1:
        data_path = './data/version_7_0latency/'
        system_info_path =  data_path + 'Mapping_usrs' + str(TOTAL_USER) +  '_coor1_latency0.txt'
        OPTIMIZE = 0


    figure_path = './figures/version_' + str(TYPE) + ' /user' + str(PLOT_USER) + '/'
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    # Get initial latency
    initial_latency = {}
    line_num = 0
    with open(system_info_path, 'r') as info_file:
        for line in info_file:
            if line_num%2 == 0:
                ele = line.split()
                user = int(ele[1])
                use_lat = LATS[int(ele[10])]
                initial_latency[user] = 20 - use_lat
                # print(use_lat, 'llllllllllll')
            line_num += 1

    ## PLot fov
    curr_user_lat = initial_latency[PLOT_USER]
    print('user', PLOT_USER, 'initial lat: ', curr_user_lat)
    fov_time, yaw, pitch = [], [], []
    fov_path = '/home/liyang/Documents/live_360/new_pickled_data/' + str(VID) +  '/' + str(PLOT_USER) + '.p'
    fov = pk.load(open(fov_path, "rb"))
    for sec_info in fov:
        if sec_info[0] >= curr_user_lat:
            for typ in sec_info[1:]:
                fov_time.append(typ[0])
                yaw.append(typ[1][0])
                pitch.append(typ[1][1])

    p = plot_fov(fov_time, yaw, pitch)
    p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_fov.eps', format='eps', dpi=1000, figsize=(20, 5))


    # Plot rate and bandwidth
    ratios = []
    ratio_path = data_path + 'ratios/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(ratio_path, 'r') as ratio_file:
        for line in ratio_file:
            ratios.append([int(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])

    p = plot_ratio(ratios)

    p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_ratio.eps', format='eps', dpi=1000, figsize=(20, 5))


    # Plot rate and bandwidth
    rates = []
    rate_path = data_path + 'rate/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(rate_path, 'r') as rate_file:
        for line in rate_file:
            rates.append([int(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])

    bw_trace = []
    bw_path = data_path + 'bw/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(bw_path, 'r') as bw_file:
        for line in bw_file:
            bw_trace.append([float(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])

    size_trace = []
    size_path = data_path + 'size/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(size_path, 'r') as size_file:
        for line in size_file:
            size_trace.append([float(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])

    p = plot_bw_rates(bw_trace, rates, size_trace)
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)
    p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_rate_bw.eps', format='eps', dpi=1000, figsize=(20, 5))




    # Plot buffer 
    buffers = []
    buffer_path = data_path + 'buffer/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(buffer_path, 'r') as buffer_file:
        for line in buffer_file:
            buffers.append([float(line.strip('\n').split(' ')[0])/Config.ms_in_s, float(line.strip('\n').split(' ')[1])/Config.ms_in_s])

    freezes = []
    freeze_path = data_path + 'freeze/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(freeze_path, 'r') as freeze_file:
        for line in freeze_file:
            freezes.append([int(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])
    
    # p = plot_freeze(freezes)
    # p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_freezes.eps', format='eps', dpi=1000, figsize=(20, 5))

    p = plot_buffer(buffers, freezes)
    p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_buffer.eps', format='eps', dpi=1000, figsize=(20, 5))

    # Plot realtime cross entropy
    entropys = []
    entropy_path = data_path + 'entropy/user' + str(PLOT_USER) + '_coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '.txt'
    with open(entropy_path, 'r') as entropy_file:
        for line in entropy_file:
            entropys.append([int(line.strip('\n').split(' ')[0]), float(line.strip('\n').split(' ')[1])])           ## < ========= value here
    
    p = plot_entropy(entropys)
    p.savefig(figure_path + 'coor' + str(COOR) + '_latency' + str(OPTIMIZE) + '_entropys.eps', format='eps', dpi=1000, figsize=(20, 5))

    #freeze
    
def plot_freeze(freezes):
    seg_offset = freezes[0][0]-1
    # y_axis_upper = max([x[1] for x in freezes])

    # if y_axis_upper >= 20:
    #     gap = 5
    # elif y_axis_upper >= 10:
    #     gap = 3
    # elif y_axis_upper >= 5:
    #     gap = 1
    # else:
    #     gap = 0.5
    # new_freezes_seg = [freezes[0][0] - seg_offset]
    # new_freezes = [freezes[0][1]]
    # for i in range(1, len(freezes)-2):
    #     new_freezes_seg.append(freezes[i][0]-seg_offset-1e-4)
    #     new_freezes.append(freezes[i-1][1])
    #     new_freezes_seg.append(freezes[i][0]-seg_offset)
    #     new_freezes.append(freezes[i][1])
    x_upper = freezes[-1][0]-seg_offset

    p = plt.figure(figsize=(20,5))
    # pl.bar(new_freezes_seg, new_freezes)
    plt.bar([x[0]-seg_offset for x in freezes], [x[1] for x in freezes], color='chocolate', label="Freeze")
    
    plt.legend(loc='upper right', fontsize=28, ncol=2)
    plt.xticks(np.arange(0, x_upper, 25), fontsize=28)
    plt.yticks(np.arange(0, 4.01, 2), fontsize=28)
    plt.ylabel('Freeze (s)', fontsize=28)
    plt.xlabel('Time (s)', fontsize=28)
    plt.axis([0, x_upper, 0, 4])
    plt.tight_layout()
    plt.close()

    return p

if __name__ == '__main__':
    main()