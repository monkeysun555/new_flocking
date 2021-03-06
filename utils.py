import numpy as np
import csv
import os 
import pickle as pk
import matplotlib.pyplot as plt
import math
from config import Config
from fastdtw import fastdtw
from collections import deque


def expand_yaw(mat, cols):
    new = []
    for v in mat:
        new += [[v for _ in range(cols)]]
    return new

def load_bw_traces(data_dir = Config.bw_trace_path):
    time_traces = []
    bw_traces = []
    datas = os.listdir(data_dir)
    for data in datas:
        if 'DS' in data:
            continue
        path = data_dir + data
        time_trace = [int(data.split('_')[3].split('.')[0])]
        bw_trace = [int(data.split('_')[3].split('.')[0])]
        with open(path) as f:
            # content = f.readlines()
            for line in f.readlines():
                info = line.strip('\n').split()
                time_trace.append(float(info[0]))
                bw_trace.append(float(info[1])*Config.bw_scale)
        time_traces.append(time_trace)
        bw_traces.append(bw_trace)
    # time_traces.sort()
    # bw_traces.sort()
    return time_traces, bw_traces

def load_bw_traces_new_5g(data_dir = Config.bw_trace_path):
    time_traces = []
    bw_traces = []
    datas = os.listdir(data_dir)
    for data in datas:
        if 'DS' in data:
            continue
        path = data_dir + data
        data_idx = int(data.split('.')[0])
        time_trace = []
        bw_trace = []
        with open(path) as f:
            # content = f.readlines()
            for line in f.readlines():
                info = line.strip('\n').split()
                time_trace.append(float(info[0]))
                bw_trace.append(float(info[1])/1000)
        # print(bw_trace)
        time_traces.append((data_idx,time_trace))
        bw_traces.append((data_idx, bw_trace))
    time_traces.sort()
    bw_traces.sort()
    return [x[1] for x in time_traces], [x[1] for x in bw_traces]

def get_tile_data_size_in_byte(rate, fps, duration=1):
    return rate*duration/fps*1000000/8/Config.n_yaw/Config.n_pitch

def get_fps():
    data_dir = Config.tsinghua_fov_data_path + 'videoMeta.csv'
    infos = []
    with open(data_dir, newline='') as csvfile:
        vidoe_info = csv.reader(csvfile)
        next(vidoe_info)
        for row in vidoe_info:
            infos.append([int(row[0]), row[2], int(row[3])])
    # print(infos)
    return infos

def translate_fov_from_tsinghua(data_dir = Config.tsinghua_fov_data_path):
    fovs = {}
    datas = os.listdir(data_dir)
    video_infos = get_fps()
    for data in datas:
        path = data_dir + data
        # print(data)
        if os.path.isdir(path):
            u_id = int(data)
            user_fovs = {}
            fov_user_datas = os.listdir(path)
            for fov_user_data in fov_user_datas:
                # video id
                video_id = int(fov_user_data.split('_')[1].split('.')[0])
                print('video id: ', video_id)
                user_fov = []
                fov_user_path = path + '/' + fov_user_data
                fps = video_infos[video_id][2]
                frame_interval = 1.0/fps
                pre_time = 0.0
                pre_floor = 0
                curr_seg = [pre_floor]
                print('frame interval: ', frame_interval)
                with open(fov_user_path, newline='') as csvfile:
                    fov_user_trace = csv.reader(csvfile)
                    next(fov_user_trace)
                    for row in fov_user_trace:
                        # print(', '.join(row))
                        # print(row)
                        # print(float(row[1]), pre_time)
                        if float(row[1]) >= pre_time:
                            euler = quaternion2euler2(row[2:6])
                            # print([x/np.pi for x in euler])
                            if np.floor(float(row[1])) > pre_floor:
                                print(curr_seg)
                                user_fov.append(curr_seg)
                                # assert np.floor(float(row[1])) == pre_floor + 1
                                pre_floor += 1
                                curr_seg = [pre_floor]
                            if len(curr_seg) <= 1 or curr_seg[-1][0] < float(row[1]):
                                curr_seg.append([float(row[1]), euler])
                            pre_time += frame_interval
                user_fov.sort()
                user_fovs[video_id] = user_fov
            fovs[u_id-1] = user_fovs 
    save_to_pickle(fovs)

def save_to_pickle(fovs):
    if not os.path.isdir(Config.pickle_root_path):
        os.mkdir(Config.pickle_root_path)

    for video_id in range(9):
        if not os.path.isdir(Config.pickle_root_path + str(video_id)):
            os.mkdir(Config.pickle_root_path + str(video_id))
        for u_id in range(48):
            print("vidoe id: ", video_id)
            print('user id: ', u_id)
            print(len(fovs[u_id][video_id]))
            pk.dump(fovs[u_id][video_id], open(Config.pickle_root_path + str(video_id) + '/' + str(u_id) + '.p', "wb" ))
    return

def show_statictic_enhanced(bw_traces):
    # trace[0] is id
    new_10 = [np.std(x[1:11])/np.mean(x[1:11]) for x in bw_traces]
    new_10.sort()
    u_len = len(new_10)//4
    print([new_10[i*u_len] for i in range(4)])

def quaternion2euler2(q):
    q1 = np.float(q[0])
    q2 = np.float(q[1])
    q3 = np.float(q[2])
    q0 = np.float(q[3])
    """convert quaternion tuple to euler angles"""
    roll = np.arctan2(2*(q0*q1+q2*q3),(1-2*(q1*q1+q2*q2)))
    # confine to [-1,1] to avoid nan from arcsin
    sintemp = min(1,2*(q0*q2-q3*q1))
    sintemp = max(-1,sintemp)
    pitch = np.arcsin(sintemp)
    yaw = np.arctan2(2*(q0*q3+q1*q2),(1-2*(q2*q2+q3*q3)))
    # assert np.abs(yaw) <= np.pi
    # assert np.abs(pitch) <= 0.5*np.pi
    return yaw, pitch, roll

def load_fovs_for_video(video_version = Config.video_version):
    fov_path = Config.pickle_root_path + str(video_version) + '/'
    video_fovs = []
    datas = os.listdir(fov_path)
    for data in datas:
        if not 'p' in data:
            continue
        path = fov_path + data
        fov = pk.load(open(path, "rb"))
        # count vertical
        # count = 0
        # for sec in fov:
        #     for frame in sec[1:]:
        #         if -0.25*np.pi <= frame[1][1] <= 0.25*np.pi:
        #             count += 1
        # print(data, count/len(fov))
        fov.insert(0, int(data.split('.')[0]))
        # for i in range(1, len(fov)):
        #     for j in range(2, len(fov[i])):
        #         if not float(fov[i][j][0]) > float(fov[i][j-1][0]):
        #             print(path)
        #             print(float(fov[i][j][0]), float(fov[i][j-1][0]))
        #             input()
        video_fovs.append(fov)
    return sorted(video_fovs), fov[-1][0]

def get_group_idx():
    latency_info = Config.latencies[Config.latency_control_version]
    group_prob_cumsum = np.cumsum(latency_info[0])
    latency_value = latency_info[1]
    l_group_idx = (group_prob_cumsum > np.random.randint(1, 100) / float(100)).argmax()
    return l_group_idx, latency_value[l_group_idx]

def get_group_idx_equal(u_id):
    latency_info = Config.latencies[Config.latency_control_version]
    # group_prob_cumsum = np.cumsum(latency_info[0])
    latency_value = latency_info[1]
    # l_group_idx = (group_prob_cumsum > np.random.randint(1, 100) / float(100)).argmax()
    l_group_idx = (u_id)%4
    buffer_upper = translate_group_to_bu_equal(l_group_idx)
    return l_group_idx, latency_value[l_group_idx], buffer_upper 

def get_group_idx_optimized(bw_trace):
    bw_trace_id = bw_trace[0]
    latency_info = Config.latencies[Config.latency_control_version]
    group_id = None
    for bw_group_idx in range(len(Config.bw_traces_group)):
        if bw_trace_id in Config.bw_traces_group[bw_group_idx]:
            group_id = bw_group_idx
            break
    assert not group_id == None
    buffer_upper = translate_group_to_bu(group_id)
    # Assume bw_group_id is the latency_id: high bw means short latency
    return group_id, latency_info[1][group_id], buffer_upper

def get_group_idx_optimized_enhanced(bw_trace):
    curr_rsd = np.std(bw_trace[1:11])/np.mean(bw_trace[1:11])   # Assume start from 1 to 11 (0 is the id)
    # get group id based on rsd of first ten seconds    # 0-3
    for g_id in reversed(range(len(Config.enhanced_rsd))):
        if curr_rsd >= Config.enhanced_rsd[g_id]:
            break
    latency_info = Config.latencies[Config.latency_control_version]
    buffer_upper = translate_group_to_bu(g_id)
    # Assume bw_group_id is the latency_id: high bw means short latency
    return g_id, latency_info[1][g_id], buffer_upper

def get_group_idx_optimized_enhanced_new_5g(trace_id):
    # get group id based on rsd of first ten seconds    # 0-3
    g_id = trace_id//int(np.ceil(Config.num_users/4))
    print(g_id)
    latency_info = Config.latencies[Config.latency_control_version]
    buffer_upper = translate_group_to_bu(g_id)
    # Assume bw_group_id is the latency_id: high bw means short latency
    return g_id, latency_info[1][g_id], buffer_upper

def translate_group_to_bu_equal(g_id):
    if g_id == 0:
        return 2000.0
    elif g_id == 1:
        return 3000.0
    elif g_id == 2:
        return 4000.0
    elif g_id == 3:
        return 5000.0
    else:
        assert 0 == 1

def translate_group_to_bu(g_id):

    # Might be used for Flocking
    # if g_id == 0:
    #     return 2000.0
    # elif g_id == 1:
    #     return 4000.0
    # elif g_id == 2:
    #     return 5000.0
    # elif g_id == 3:
    #     return 6000.0
    # else:
    #     assert 0 == 1

    # Modified for TMM
    if g_id == 0:
        return 3000.0
    elif g_id == 1:
        return 4000.0
    elif g_id == 2:
        return 5000.0
    elif g_id == 3:
        return 6000.0
    else:
        assert 0 == 1

def similarity(x,y):
    yaw_distance = min(np.abs(x[0] - y[0]), 2*np.pi - np.abs(x[0] - y[0]))
    pitch_distance = np.abs(x[1]-y[1])
    return np.sqrt(yaw_distance**2+pitch_distance**2)

def degree_similarity(x,y):
    yaw_distance = min(np.abs(x[0] - y[0]), 360.0 - np.abs(x[0] - y[0]))
    pitch_distance = np.abs(x[1]-y[1])
    return np.sqrt(yaw_distance**2+pitch_distance**2)

def sphere_distance(x,y):
    lon1 = x[0]/180.0*np.pi   # yaw
    lat1 = x[1]/180.0*np.pi   # pitch
    lon2 = y[0]/180.0*np.pi
    lat2 = y[1]/180.0*np.pi  
    delta = math.acos(max(-1.0, min(1.0, math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2) + math.sin(lat1)*math.sin(lat2))))
    return delta

def sphere_distance_new(x,y):
    lon1 = x[0]
    lat1 = x[1]
    lon2 = y[0]
    lat2 = y[1]
    delta = math.acos(max(-1.0, min(1.0, math.cos(lat1)*math.cos(lat2)*math.cos(lon1-lon2) + math.sin(lat1)*math.sin(lat2))))
    return delta

def transform_similarity(x, y):
    # Attention: if add weight, the true value change, adjust epsilon accordingly
    yaw_distance = min(np.abs(x[0] - y[0]), 2*np.pi - np.abs(x[0] - y[0]))
    pitch_distance = np.abs(x[1]-y[1])
    return np.sqrt((2.0*Config.n_pitch*yaw_distance/Config.n_yaw)**2+pitch_distance**2)

def truncated_linear(gap, time_trace, Xs):
    # The data in Xs is already shifted for kf, and the Xs is modified by kf
    # assert len(time_trace)-2 == len(Xs)
    yaw_trace = [x[0] for x in Xs]
    pitch_trace = [x[1] for x in Xs]
    # yaw_diff = np.diff(yaw_trace)
    # pitch_diff = np.diff(pitch_trace)
    # print("Max of yaw diff: ", max(yaw_diff))
    # print("Max fo pitch diff: ", max(pitch_diff))
    # Truncate trace
    truncated_yaw_time, truncated_yaw = truncate_trace(time_trace[2:], yaw_trace)
    truncated_pitch_time, truncated_pitch = truncate_trace(time_trace[2:], pitch_trace)
    # Linear prediction
    # print(truncated_yaw_time, truncated_yaw)
    # input()
    yaw_predict_model = np.polyfit(truncated_yaw_time, truncated_yaw, Config.trun_regression_order)
    yaw_predict_value = np.round(np.polyval(yaw_predict_model, gap))
    pitch_predict_model = np.polyfit(truncated_pitch_time, truncated_pitch, Config.trun_regression_order)
    pitch_predict_value = np.round(np.polyval(pitch_predict_model, gap))
    return (yaw_predict_value%360.0, min(180, max(pitch_predict_value, 0)))

def truncated_linear_new(gaps, time_trace, Xs):
    # The data in Xs is already shifted for kf, and the Xs is modified by kf
    # assert len(time_trace)-2 == len(Xs)
    yaw_trace = [x[0] for x in Xs]
    pitch_trace = [x[1] for x in Xs]
    # yaw_diff = np.diff(yaw_trace)
    # pitch_diff = np.diff(pitch_trace)
    # print("Max of yaw diff: ", max(yaw_diff))
    # print("Max fo pitch diff: ", max(pitch_diff))
    # Truncate trace
    truncated_yaw_time, truncated_yaw = truncate_trace(time_trace[2:], yaw_trace)
    truncated_pitch_time, truncated_pitch = truncate_trace(time_trace[2:], pitch_trace)
    yaw_predict_model = np.polyfit(truncated_yaw_time, truncated_yaw, Config.trun_regression_order)
    pitch_predict_model = np.polyfit(truncated_pitch_time, truncated_pitch, Config.trun_regression_order)
    # Linear prediction
    # print(truncated_yaw_time, truncated_yaw)
    # input()
    centers = []
    for i in range(len(gaps)):
        sec_centers = []
        for gap in gaps[i]:
            yaw_predict_value = np.round(np.polyval(yaw_predict_model, gap))
            pitch_predict_value = np.round(np.polyval(pitch_predict_model, gap))
            sec_centers += [(yaw_predict_value%360.0, min(180, max(pitch_predict_value, 0)))]
        centers += [sec_centers]
    return centers

def calculate_curve_distance(c1, c2):
    # assert len(c1) == len(c2)
    # distance, path = fastdtw(c1, c2, dist=degree_similarity)
    distance, path = fastdtw(c1, c2, dist=sphere_distance)
    return distance, path

def truncate_trace(time_trace, trace):
    # for testing
    # return time_trace, trace
    # assert len(time_trace) == len(trace)
    tail_idx = len(trace)-2
    # print(len(trace))
    # assert len(trace) >= Config.predict_tth_len
    while tail_idx >=0 and trace[tail_idx] == trace[tail_idx+1]:
        tail_idx -= 1
    if tail_idx < 0 :
        return time_trace, trace
    current_sign = np.sign(trace[tail_idx+1] - trace[tail_idx])     # Get real sign, in order

    # If 0 range is large, no linear
    if len(trace) - tail_idx >= Config.zero_prediction_tth:
        return time_trace[-Config.zero_prediction_tth:], trace[-Config.zero_prediction_tth:]
    else:
        # Truncate trace
        while tail_idx > 0:    
            if np.sign(trace[tail_idx+1] - trace[tail_idx]) == current_sign or abs(trace[tail_idx+1] - trace[tail_idx]) <= 1:
                tail_idx -= 1
            else:
                break
    truncated_trace = trace[tail_idx+1:]
    trincated_time = time_trace[tail_idx+1:]
    return trincated_time, truncated_trace

def get_time_weight(gap_in_s):
    # print(gap_in_s)
    return np.e**(-0.5*(gap_in_s))
    # return 1.0
    
def tile_cross_entropy(ground_truth, prediction, add = 0):
    # ground_truth, prediction = prediction, ground_truth
    # assert np.round(np.sum(ground_truth), 2) == 1
    # assert np.round(np.sum(prediction), 2) == 1
    prediction = np.clip(prediction, 1e-12, 1. - 1e-12)
    ground_truth = np.clip(ground_truth, 1e-12, 1. - 1e-12)
    entropy = -np.sum(ground_truth*np.log(prediction)) - (-np.sum(ground_truth*np.log(ground_truth)))
    entropy += add
    return entropy

def calculate_ratios(ground_truth, prediction):
    ratio = 0
    for i in range(len(ground_truth)):
        for j in range(len(ground_truth[i])):
            ratio += min(ground_truth[i][j], prediction[i][j])
    return ratio

def get_weight(distance):
    return 1 - 1/(1+np.e**(-Config.p1*(distance-Config.p2)))

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

def get_tile_overlap_ratio(target_gt_dis, pred_dis):
    watched_tiles = set()
    for i in range(len(target_gt_dis)):
        for j in range(len(target_gt_dis[i])):
            if target_gt_dis[i][j] > 0:
                watched_tiles.add((i,j))

    pred_tiles = []
    for i in range(len(pred_dis)):
        for j in range(len(pred_dis[i])):
            pred_tiles.append((pred_dis[i][j], i, j))

    if len(pred_tiles) < len(watched_tiles):
        return len(pred_tiles)/len(watched_tiles)
    pred_tiles.sort(key = lambda x: -x[0])
    count = 0
    for i in range(len(watched_tiles)):
        [x,y] = pred_tiles[i][1:3]
        if (x,y) in watched_tiles:
            count += 1
    assert count <= len(watched_tiles)
    return count/len(watched_tiles)
        
class kalman_filter(object):
    def __init__(self):
        self.time_trace = None
        self.yaw_trace = None
        self.pitch_trace = None
        self.Q_v = 0.1
        self.R_v = 0.1

    def set_traces(self, time_trace, yaw_trace, pitch_trace):
        self.time_trace = time_trace
        self.yaw_trace = yaw_trace
        self.pitch_trace = pitch_trace
        self.modified_kf_x = []
        self.kf_x = []

    def init_kf(self):
        self.B = 0;
        self.U = 0;
        self.P = np.diag((0.01, 0.01, 0.01, 0.01))
        self.dt = self.time_trace[1] - self.time_trace[0]
        # assert self.dt > 0 
        self.dx = self.yaw_trace[1] - self.yaw_trace[0]
        self.dy = self.pitch_trace[1] - self.pitch_trace[0]
        self.X = np.array([[self.yaw_trace[1]],
                          [self.pitch_trace[1]],
                          [self.dx/self.dt], 
                          [self.dy/self.dt]])
        # self.X = np.array([[self.yaw_trace[1]],
        #                   [self.pitch_trace[1]],
        #                   [0.0], 
        #                   [0.0]])

        self.A = np.array([[1, 0 , self.dt, 0],
                           [1, 0 , 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Assume measurement does NOT inlucde velocity x/y
        # self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])     # 2*4
        # self.Y = np.array([[0.0], [0.0]])                   # Will udpate later
        # self.Q = np.eye(self.X.shape[0])*self.Q_v           # 4*4
        # self.R = np.eye(self.Y.shape[0])*self.R_v           # 2*2

        # Include vx and vy
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])     # 2*4
        self.Y = np.array([[0.0], [0.0], [0.0], [0.0]])                   # Will udpate later
        self.Q = np.eye(self.X.shape[0])*self.Q_v           # 4*4
        self.R = np.eye(self.Y.shape[0])*self.R_v           # 2*2

    def kf_update(self):
        IM = np.dot(self.H, self.X)                                 # 2*1
        IS = self.R + np.dot(self.H, np.dot(self.P, self.H.T))      # 2*2
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(IS)))     # 4*2
        self.X = self.X + np.dot(K, (self.Y-IM))                    # 4*1
        self.P = self.P - np.dot(K, np.dot(IS, K.T))                # 4*4
        # LH = gauss_pdf(Y, IM, IS) 
        return (K,IM,IS)

    def kf_predict(self):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.U)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def kf_update_para(self, i):
        self.dt = self.time_trace[i] - self.time_trace[i-1]
        self.dx = self.yaw_trace[i] - self.yaw_trace[i-1]
        self.dy = self.pitch_trace[i-1] - self.pitch_trace[i-1]
        if not self.dt == 0:
            # Assume there is no vx/vy
            # self.Y = np.array([[self.yaw_trace[i]], 
            #                    [self.pitch_trace[i]]])

            # There are vx/vy
            self.Y = np.array([[self.yaw_trace[i]], 
                               [self.pitch_trace[i]],
                               [self.dx/self.dt],
                               [self.dy/self.dt]])
            self.A = np.array([[1.0, 0 , self.dt, 0],
                                [0, 1.0 , 0, self.dt],
                                [0, 0, 1.0, 0],
                                [0, 0, 0, 1.0]])

    def kf_run(self, time_gap = 0.5, kf_predict=False):
        # return [[self.yaw_trace[i], self.pitch_trace[i]] for i in range(2, len(self.time_trace))]
        for i in range(2, len(self.time_trace)):
            self.kf_update_para(i)
            self.kf_predict()
            # Get predict info before slef.x is updated using measurement Y
            self.kf_x.append(self.X.T[0][:2])
            self.kf_update()
            self.modified_kf_x.append(self.X.T[0][:2])

        if Config.show_kf:
            plt.scatter(self.time_trace[2:], [p for p in self.yaw_trace[2:]], c='r', s=20)
            # plt.scatter(self.time_trace[2:], [p[0] for p in self.kf_x], c='b')
            plt.scatter(self.time_trace[2:], [p[0] for p in self.modified_kf_x], c='g', s=20)
            plt.show()
            input()

            plt.scatter(self.time_trace[2:], [p for p in self.pitch_trace[2:]], c='r', s=20)
            # plt.scatter(self.time_trace[2:], [p[1] for p in self.kf_x], c='b')
            plt.scatter(self.time_trace[2:], [p[1] for p in self.modified_kf_x], c='g', s=20)
            plt.show()
            input()

        if kf_predict:
            new_A = np.array([[1, 0 , time_gap, 0],
                              [0, 1 , 0, time_gap],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
            new_Xs = np.dot(new_A, self.X).tolist()
            return new_Xs
        else:
            return self.modified_kf_x



def main():
    translate_fov_from_tsinghua()
    # a = load_fovs_for_video()
    # print(a)

if __name__ == '__main__':
    main()
