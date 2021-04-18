import os
import numpy as np
import utils
import queue as Q
from config import Config
from sklearn.cluster import KMeans
from load_models import *
from scipy.io import loadmat
from multiprocessing import Lock, Process, Manager


prediction_types = [2,4]        # 1: LSTM_t 2: LSTM_c 3: TLP 4: co-TLP

num_of_others = [6, 18, 30]
seg_duration = 1000.0
video_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
# video_list = [40]
num_process = 6
num_others = 30

def process_degree(trace):
        # print(trace)
        new_trace = [trace[0]]
        for value in trace[1:]:
            if np.abs(value - new_trace[-1]) > 180.0:
                if value < new_trace[-1]:
                    new_trace.append(value+360.0)
                else:
                    new_trace.append(value-360.0)
            else:
                new_trace.append(value)
        return new_trace

def select_users(users, target_len):
    while len(users) > target_len:
        users.pop()
    return users
    
def main():
    
    processes = []
    users_per_process = num_others//num_process
    for i in range(num_process):
        curr_range = [k for k in range(i*users_per_process, (i+1)*users_per_process)]
        processes.append(Process(target=multi_processes, args=(curr_range, i, video_list)))
        processes[-1].start()
                # processes[-1].join()
    for process in processes:
        #   """
        #   Waits for threads to complete before moving on with the main
        #   script.
        #   """
        process.join()


    type_datas = {}
    # collect results
    # for vid in video_list:
    for range_id in range(num_process):
        res_path = './fov_prediction/numbers_range' + str(range_id) + '.txt'
        with open(res_path, 'r') as fr:
            line_number = 0
            for line in fr:
                line = line.split(' ')
                if line_number == 0 or line_number == 16:
                    t = int(line[0])
                    counter = 0
                else:
                    if t not in type_datas:
                        type_datas[t] = [[[] for _ in range(5)] for _ in range(3)]
                    print(counter)
                    number_idx = counter//5
                    interval = counter%5
                    entropy = float(line[0])
                    type_datas[t][number_idx][interval] += [entropy]
                    counter += 1
                line_number += 1



    log_path = './fov_prediction/number.txt'

    accuracy_write = open(log_path, 'w')

    for key, val in type_datas.items():
        print("type ", key)
        for j in range(3):
            print("number idx: ", j)
            for k in range(5):
                print(np.mean(val[j][k]))
                accuracy_write.write(str(np.mean(val[j][k])) + ' ')
            accuracy_write.write('\n')


    # NSS
    type_datas = {}
    # collect results
    # for vid in video_list:
    for range_id in range(num_process):
        res_path = './fov_prediction/numbers_nss_range' + str(range_id) + '.txt'
        with open(res_path, 'r') as fr:
            line_number = 0
            for line in fr:
                line = line.split(' ')
                if line_number == 0 or line_number == 16:
                    t = int(line[0])
                    counter = 0
                else:
                    if t not in type_datas:
                        type_datas[t] = [[[] for _ in range(5)] for _ in range(3)]
                    print(counter)
                    number_idx = counter//5
                    interval = counter%5
                    entropy = float(line[0])
                    type_datas[t][number_idx][interval] += [entropy]
                    counter += 1
                line_number += 1



    log_path = './fov_prediction/number_nss.txt'

    accuracy_write = open(log_path, 'w')

    for key, val in type_datas.items():
        print("type ", key)
        for j in range(3):
            print("number idx: ", j)
            for k in range(5):
                print(np.mean(val[j][k]))
                accuracy_write.write(str(np.mean(val[j][k])) + ' ')
            accuracy_write.write('\n')

   


def multi_processes(user_range, range_id, video_range):
    pred_interval = np.ones((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))
    for i in range(Config.target_user_pred_len):
        pred_interval[i] *= (i+1)/Config.target_user_pred_len/Config.n_pitch/Config.n_yaw
    pred_interval = np.expand_dims(pred_interval, axis=0)

    pred_interval_new = np.ones((Config.target_user_pred_len,1))
    for i in range(Config.target_user_pred_len):
        pred_interval_new[i] *= (i+1)/Config.target_user_pred_len
    pred_interval_new = np.expand_dims(pred_interval_new, axis=0)

    tile_map = loadmat(Config.tile_map_dir_new)['map']

    kf = utils.kalman_filter()

    model = load_decay_model_new()
    target_model = load_keras_target_model()
    print("load model successfully!")

    all_loss = {}
    all_nss = {}

    for t in prediction_types:
        all_loss[t] = []
        all_nss[t] = []

    for t in prediction_types:  
        for vid in video_range:
            nums_losses = [[] for _ in range(len(num_of_others))]
            nums_nsses = [[] for _ in range(len(num_of_others))]

            video_fovs, v_length = utils.load_fovs_for_video(vid)        # (48, *) fovs for one video

            for nums_id in range(len(num_of_others)):
                nums = num_of_others[nums_id]
                losses = [[] for _ in range(5)]
                nsses = [[] for _ in range(5)]

                for i in user_range:
                    print('user:', i)
                    user_fov = video_fovs[i][1:]

                    # select other users
                    other_users = [uid for uid in range(num_others) if uid != i]
                    other_users = select_users(other_users, nums)

                    if t == 4:
                        for seg_id in range(1,len(user_fov)-10):
                            past_second_fov = user_fov[seg_id][1:]
                            # The time is from 0 - 1, offset is removed
                            time_trace, yaw_trace, pitch_trace = [x[0] for x in past_second_fov], [x[1][0]/np.pi*180.0+180 for x in past_second_fov], [x[1][1]/np.pi*180.0+90 for x in past_second_fov]
                            # Has to process yaw_trace
                            processed_yaw_trace = process_degree(yaw_trace)
                            # print(time_trace, processed_yaw_trace, pitch_trace)

                            # Get current seg similarity
                            # Calculate similarity
                            future_other_ave = [np.zeros((Config.n_pitch, Config.n_yaw)) for _ in range(5)]
                            current_user_seg_in_pi = [(processed_yaw_trace[jjj]/180*np.pi, pitch_trace[jjj]/180*np.pi) for jjj in range(len(yaw_trace))]
                            similarity = [0]*len(other_users)
                            for jj in range(len(other_users)):
                                other_fov = video_fovs[other_users[jj]][1:]
                                other_past_seg_fov = other_fov[seg_id][1:]

                                # Calculate distance
                                other_yaw, other_pitch = [x[1][0]/np.pi*180.0+180 for x in other_past_seg_fov], [x[1][1]/np.pi*180.0+90 for x in other_past_seg_fov]
                                other_yaw_processed = process_degree(other_yaw)
                                other_seg_in_pi = [(other_yaw_processed[jjj]/180*np.pi, other_pitch[jjj]/180*np.pi) for jjj in range(len(other_yaw_processed))]
                                min_len = min(len(current_user_seg_in_pi), len(other_seg_in_pi))
                                distance = utils.calculate_curve_distance(current_user_seg_in_pi[:min_len], other_seg_in_pi[:min_len])[0]
                                # print(distance)
                                similarity[jj] = utils.get_weight(distance)
                                # print(similarity)
                                
                            total_weight = np.sum(similarity)

                            for jj in range(len(other_users)):
                                other_fov = video_fovs[other_users[jj]][1:]
                                for jjj in range(5):
                                    future_seg_id = seg_id + jjj + 1
                                    other_future_seg_fov = other_fov[future_seg_id][1:]
                                    other_future_distributions = get_distribution_from_center(other_future_seg_fov, tile_map)

                                    future_other_ave[jjj] += other_future_distributions*similarity[jj]/total_weight

                            for jjj in range(5):
                                future_other_ave[jjj] /= np.sum(future_other_ave[jjj])

                            kf.set_traces(time_trace, processed_yaw_trace, pitch_trace)
                            kf.init_kf()

                            modified_Xs = kf.kf_run()

                            # get gap for future 5 seconds
                            future_second_times = []
                            last_frame_time = time_trace[-1]
                            second_gaps = []
                            for future_id in range(5):
                                prediction_gap = last_frame_time + future_id
                                gaps = []
                                interval = 1/(Config.num_interval+1)
                                for frame_in_second in range(Config.num_interval):
                                    gaps += [(prediction_gap+(i+1)*interval)]
                                second_gaps += [gaps]
                            # num_interval = Config.num_interval

                            centers = utils.truncated_linear_new(second_gaps, time_trace, modified_Xs)
                            assert len(centers) == 5
                            # print(centers)

                            seg_distributions = []
                            for ii in range(5):
                                absolute_seg_id = seg_id + ii + 1
                                seg_frames = centers[ii]
                                curr_distributions = np.zeros((Config.n_pitch, Config.n_yaw))
                                # To calculate accuracy
                                for center in seg_frames:
                                    curr_distributions += tile_map[int(center[1])][int(center[0])]/np.sum(tile_map[int(center[1])][int(center[0])])
                                curr_distributions /= np.sum(curr_distributions)

                                # Collaborate  using future_other_ave (which is the weighted-ave for others in 5 future seconds)
                                target_self_weight = max(1/(1+total_weight), 0.8)
                                assert target_self_weight <= 1
                                final_dis = np.multiply(curr_distributions, target_self_weight) + np.multiply(future_other_ave[ii], 1-target_self_weight)
                                
                                final_dis /= np.sum(final_dis)

                                # Then calculate gt for target user
                                future_second_fov = user_fov[absolute_seg_id][1:]
                                target_gt_dis = get_distribution_from_center(future_second_fov, tile_map)

                                cross_entropy = utils.tile_cross_entropy(target_gt_dis, final_dis) #  TMM 0.5

                                new_ratio = get_tile_overlap_ratio(target_gt_dis,final_dis)

                                losses[ii].append(cross_entropy)

                                # normalized = (final_dis - np.mean(final_dis))/np.std(final_dis)
                                # scanpath, total_points = get_scanpath_from_center(future_second_fov, tile_map)
                                # nss_score = np.sum(normalized * scanpath)/total_points
                                
                                nsses[ii].append(new_ratio)
                                # nsses[ii].append(nss_score)

                    # if t == 2
                    else:
                        for seg_id in range(1,len(user_fov)-10):
                            target_prev_dis = []
                            for sec_id in range(seg_id - Config.target_his_fov_len, seg_id):
                                seg_trace = user_fov[sec_id][1:]
                                seg_dis = get_distribution_from_center(seg_trace, tile_map)
                                target_prev_dis += [seg_dis]

                            target_prev_dis = np.array(target_prev_dis)
                            the_last_seg_dis = np.expand_dims(target_prev_dis[-1], axis=0)
                            target_prev_dis = np.expand_dims(target_prev_dis, axis=0)
                            the_last_seg_dis = np.expand_dims(the_last_seg_dis, axis=0)

                            
                            # if t == 1:
                            #     output = target_model.predict([target_prev_dis, the_last_seg_dis, pred_interval])
                            # else:
                            # prepare other mean
                            future_other_ave = [np.zeros((Config.n_pitch, Config.n_yaw)) for _ in range(5)]
                            future_other_var = [np.zeros((Config.n_pitch, Config.n_yaw)) for _ in range(5)]
                            distirbutions = [[] for _ in range(5)]
                            for jjj in range(5):
                                future_seg_id = seg_id + jjj + 1
                                for jj in range(len(other_users)):

                                    other_fovs = video_fovs[other_users[jj]][1:]
                                    other_future_seg_fov = other_fovs[future_seg_id][1:]
                                    other_future_distributions = get_distribution_from_center(other_future_seg_fov, tile_map)
                                    distirbutions[jjj] += [other_future_distributions]
                                    # future_other_ave[jjj] += other_future_distributions
                            distirbutions = np.array(distirbutions)
                            for jjj in range(5):
                                future_other_ave[jjj] = np.mean(distirbutions[jjj], axis = 0)
                                future_other_var[jjj] = np.std(distirbutions[jjj], axis = 0)

                            # future_other_ave = np.array(future_other_ave)
                            num_other = np.ones((Config.target_user_pred_len,1))*(num_others/31)
                            num_other = np.expand_dims(num_other, axis=0)

                            future_other_ave = np.expand_dims(future_other_ave, axis=0)
                            future_other_var = np.expand_dims(future_other_ave, axis=0)
                            # other_std = np.expand_dims(other_std, axis=0)
                            future_other_ave = np.expand_dims(future_other_ave, axis=4)
                            future_other_var = np.expand_dims(future_other_ave, axis=4)

                            output = model.predict([target_prev_dis, the_last_seg_dis, future_other_ave, future_other_var, num_other, pred_interval_new])

                            output = output[0].reshape((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))
                            for ii in range(5):
                                absolute_seg_id = seg_id + ii + 1
                                future_second_fov = user_fov[absolute_seg_id][1:]
                                target_gt_dis = get_distribution_from_center(future_second_fov, tile_map)
                                cross_entropy = utils.tile_cross_entropy(target_gt_dis, output[ii])
                                losses[ii].append(cross_entropy)


                                new_ratio = get_tile_overlap_ratio(target_gt_dis,output[ii])

                                nsses[ii].append(new_ratio)

                                # final_dis = output[ii]
                                # normalized = (final_dis - np.mean(final_dis))/np.std(final_dis)
                                # # nss
                                # scanpath, total_points = get_scanpath_from_center(future_second_fov, tile_map)
                                # nss_score = np.sum(normalized * scanpath)/total_points
                                # nsses[ii].append(nss_score)

                    # else:
                    #     # for seg_id in range(4,len(user_fov)):
                    #         # pass
                    #     pass
                nums_losses[nums_id] = losses
                nums_nsses[nums_id] = nsses

            all_loss[t] += [nums_losses]
            all_nss[t] += [nums_nsses]

    new_format = {}
    for key, value in all_loss.items():
        new_format[key] = [[[] for _ in range(5)] for _ in range(3)]    # Three value of number of others
        # key is type of prediction, value is all data
        print('type is:', key)
        print(len(value))
        curr_type_data = value
        for vid in range(len(value)):
            curr_video_data = value[vid]

            print('vid: ', vid)
            print(len(curr_video_data))
            for num_id in range(len(num_of_others)):
                curr_datas = curr_video_data[num_id]   # losses including 5 intervals, in each are the data of user and seg
                print(len(curr_datas))
                # there are five in each
                for interval in range(5):
                    print(curr_datas[interval])
                    new_format[key][num_id][interval] += curr_datas[interval]

    new_format_nss = {}
    for key, value in all_nss.items():
        new_format_nss[key] = [[[] for _ in range(5)] for _ in range(3)]    # Three value of number of others
        # key is type of prediction, value is all data
        print('type is:', key)
        print(len(value))
        curr_type_data = value
        for vid in range(len(value)):
            curr_video_data = value[vid]

            print('vid: ', vid)
            print(len(curr_video_data))
            for num_id in range(len(num_of_others)):
                curr_datas = curr_video_data[num_id]   # losses including 5 intervals, in each are the data of user and seg
                print(len(curr_datas))
                # there are five in each
                for interval in range(5):
                    print(curr_datas[interval])
                    new_format_nss[key][num_id][interval] += curr_datas[interval]


    if not os.path.isdir('./fov_prediction/'):
        os.makedirs('./fov_prediction/')
    log_path = './fov_prediction/numbers_range' + str(range_id) + '.txt'

    accuracy_write = open(log_path, 'w')

    for key, value in new_format.items():
        print("type: ", key)
        accuracy_write.write(str(key) + ' ')
        accuracy_write.write('\n')

        for i in range(len(num_of_others)):
            print('number is: ', num_of_others[i]) 
            curr_number_values = value[i]
            for interval in range(len(curr_number_values)):
                print('INterval ', interval)
                print("kl: ", np.mean(curr_number_values[interval]))

                accuracy_write.write(str(np.mean(curr_number_values[interval])) + ' ')
                accuracy_write.write('\n')


    log_path = './fov_prediction/numbers_nss_range' + str(range_id) + '.txt'

    nss_write = open(log_path, 'w')

    for key, value in new_format_nss.items():
        print("type: ", key)
        nss_write.write(str(key) + ' ')
        nss_write.write('\n')

        for i in range(len(num_of_others)):
            print('number is: ', num_of_others[i]) 
            curr_number_values = value[i]
            for interval in range(len(curr_number_values)):
                print('INterval ', interval)
                print("nss: ", np.mean(curr_number_values[interval]))

                nss_write.write(str(np.mean(curr_number_values[interval])) + ' ')
                nss_write.write('\n')
    


def get_distribution_from_center(frames_info, tile_map):
    # This is for all frames within a segment
    n_frames = float(len(frames_info))
    frame_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
    for f_info in frames_info:
        pitch_center = int(np.round(f_info[1][1]/np.pi*180)) + 90        # Index offset
        yaw_center = int(np.round(f_info[1][0]/np.pi*180)) + 180         # Index offset
        # print(pitch_center, yaw_center)
        tiles = tile_map[pitch_center][yaw_center]
        frame_distribution += np.array(tiles)/np.sum(tiles)     # Get total numbers for tiles for one frame, then normalized
    frame_distribution /= n_frames
    return frame_distribution

def get_scanpath_from_center(frames_info, tile_map):
    total_points = 0
    scanpath = np.zeros((Config.n_pitch, Config.n_yaw))
    for frame_info in frames_info:
        gt_yaw, gt_pitch = int((frame_info[1][0]/np.pi*180.0+180.0)%360), int(frame_info[1][1]/np.pi*180.0+90.0)
        tiles = tile_map[gt_pitch][gt_yaw]
        # row_idx, col_idx = int(gt_pitch/180*Config.n_pitch), int(gt_yaw/360*Config.n_yaw)
        binary_map = np.where(tiles > 0.5, 1, 0)
        scanpath += binary_map
        total_points += np.sum(binary_map)
    # frame_distribution /= n_frames
    return scanpath, total_points

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
    
if __name__ == '__main__':
    main()