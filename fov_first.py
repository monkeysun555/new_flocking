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
num_others = 30

# data_keys = [x for x in range(33, 56)] This is validation dataset
# Choose some for testing
video_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
# video_list = [49]

random_step_id = [x for x in range(10, 50)]
target_vid = 49
target_user = 23
seg_duration = 1000.0
num_process = 6


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
        users.pop(0)
    return users
    
def main():

    if not os.path.isdir('./fov_prediction/pred_dis/'):
        os.makedirs('./fov_prediction/pred_dis/')

    processes = []
    users_per_process = num_others//num_process
    for i in range(num_process):
        curr_range = [k for k in range(i*users_per_process, (i+1)*users_per_process)]
        processes.append(Process(target=multi_processes, args=(curr_range, i, video_list)))
        processes[-1].start()
                # processes[-1].join()
    for process in processes:
        process.join()

    type_datas = {}
    # collect results
    for vid in video_list:
        for range_id in range(num_process):
            res_path = './fov_prediction/accuracy_first' + str(range_id) + '.txt'
            with open(res_path, 'r') as fr:
                t = 2
                for line in fr:
                    line = line.split(' ')
                    if t not in type_datas:
                        type_datas[t] = [[] for _ in range(1)]
                    for j in range(1):
                        type_datas[t][j] += [float(line[j])]
                    t += 2

    log_path = './fov_prediction/accuracy_first_all.txt'

    accuracy_write = open(log_path, 'w')

    for key, val in type_datas.items():
        print("type ", key)
        for j in range(1):
            print(np.mean(val[j]))
            accuracy_write.write(str(np.mean(val[j])) + ' ')
        accuracy_write.write('\n')


    ## NSS
    # type_datas_nss = {}
    # # collect results
    # for vid in video_list:
    #     for range_id in range(num_process):
    #         res_path = './fov_prediction/nss_range' + str(range_id) + '.txt'
    #         with open(res_path, 'r') as fr:
    #             t = 1
    #             for line in fr:
    #                 line = line.split(' ')
    #                 if t not in type_datas_nss:
    #                     type_datas_nss[t] = [[] for _ in range(5)]
    #                 for j in range(5):
    #                     type_datas_nss[t][j] += [float(line[j])]
    #                 t += 1

    # log_path = './fov_prediction/nss.txt'
    # nss_write = open(log_path, 'w')
    # for key, val in type_datas_nss.items():
    #     print("type ", key)
    #     for j in range(5):
    #         print(np.mean(val[j]))
    #         nss_write.write(str(np.mean(val[j])) + ' ')
    #     nss_write.write('\n')


def multi_processes(user_range, range_id, video_range):
    print(user_range)
    intervals = Config.first_prediction_interval
    pred_interval = np.ones((intervals, Config.n_pitch, Config.n_yaw))
    for i in range(intervals):
        pred_interval[i] *= (i+1)/intervals/Config.n_pitch/Config.n_yaw
    pred_interval = np.expand_dims(pred_interval, axis=0)

    pred_interval_new = np.ones((intervals,1))
    for i in range(intervals):
        pred_interval_new[i] *= (i+1)/intervals
    pred_interval_new = np.expand_dims(pred_interval_new, axis=0)

    tile_map = loadmat(Config.tile_map_dir_new)['map']
    model = load_decay_first()  # to new
    target_model = load_keras_target_model()
    print("load model successfully!")
    all_loss = {}
    all_nss = {}
    for t in [2,4]:
        all_loss[t] = []
        all_nss[t] = []


    for t in prediction_types:  
        # loss_types = []

        for vid in video_range:
            losses = [[] for _ in range(intervals)]
            nsses = [[] for _ in range(intervals)]
            video_fovs, v_length = utils.load_fovs_for_video(vid)        # (48, *) fovs for one video
            # print(video_fovs, v_length)
            kf = utils.kalman_filter()
            for i in user_range:
                print('user:', i)
                user_fov = video_fovs[i][1:]

                # select other users
                other_users = [uid for uid in range(31) if uid != i]
                other_users = select_users(other_users, num_others)

                if t in [3,4]:
                    for seg_id in range(1,len(user_fov)-10):
                        past_second_fov = user_fov[seg_id][1:]
                        # The time is from 0 - 1, offset is removed
                        time_trace, yaw_trace, pitch_trace = [x[0] for x in past_second_fov], [x[1][0]/np.pi*180.0+180 for x in past_second_fov], [x[1][1]/np.pi*180.0+90 for x in past_second_fov]
                        # Has to process yaw_trace
                        processed_yaw_trace = process_degree(yaw_trace)
                        # print(time_trace, processed_yaw_trace, pitch_trace)

                        # Get current seg similarity
                        # Calculate similarity
                        if t == 4:
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
                                for jjj in range(intervals):
                                    future_seg_id = seg_id + jjj + 1
                                    other_future_seg_fov = other_fov[future_seg_id][1:]
                                    other_future_distributions = get_distribution_from_center(other_future_seg_fov, tile_map)

                                    future_other_ave[jjj] += other_future_distributions*similarity[jj]/total_weight

                            for jjj in range(intervals):
                                future_other_ave[jjj] /= np.sum(future_other_ave[jjj])

                        kf.set_traces(time_trace, processed_yaw_trace, pitch_trace)
                        kf.init_kf()

                        modified_Xs = kf.kf_run()

                        # get gap for future 5 seconds
                        future_second_times = []
                        last_frame_time = time_trace[-1]
                        second_gaps = []
                        for future_id in range(intervals):
                            prediction_gap = last_frame_time + future_id
                            gaps = []
                            interval = 1/(Config.num_interval+1)
                            for frame_in_second in range(Config.num_interval):
                                gaps += [(prediction_gap+(i+1)*interval)]
                            second_gaps += [gaps]
                        # num_interval = Config.num_interval

                        centers = utils.truncated_linear_new(second_gaps, time_trace, modified_Xs)
                        # assert len(centers) == 5
                        # print(centers)

                        seg_distributions = []
                        for ii in range(intervals):
                            absolute_seg_id = seg_id + ii + 1
                            seg_frames = centers[ii]
                            curr_distributions = np.zeros((Config.n_pitch, Config.n_yaw))
                            # To calculate accuracy
                            for center in seg_frames:
                                curr_distributions += tile_map[int(center[1])][int(center[0])]/np.sum(tile_map[int(center[1])][int(center[0])])
                            curr_distributions /= np.sum(curr_distributions)

                            if t == 4:
                                # Collaborate  using future_other_ave (which is the weighted-ave for others in 5 future seconds)
                                target_self_weight = max(1/(1+total_weight), 0.8)
                                assert target_self_weight <= 1
                                final_dis = np.multiply(curr_distributions, target_self_weight) + np.multiply(future_other_ave[ii], 1-target_self_weight)
                            else:
                                final_dis = curr_distributions
                            final_dis /= np.sum(final_dis)


                            # Then calculate gt for target user
                            future_second_fov = user_fov[absolute_seg_id][1:]
                            target_gt_dis = get_distribution_from_center(future_second_fov, tile_map)

                            cross_entropy = utils.tile_cross_entropy(target_gt_dis, final_dis) #  TMM 0.5
                            
                            # if vid == target_vid and i == target_user and seg_id in random_step_id:
                            #     # find the path
                            #     dis_file_path = './fov_prediction/pred_dis/' + str(seg_id) + '.txt'
                            #     file1 = open(dis_file_path, "a")  # append mode 
                            #     print(t, ii, seg_id)
                            #     file1.write(str(t) + '\t' + str(ii) +  '\n') 
                            #     for rr in final_dis:
                            #         for cc in rr:
                            #             file1.write( str(cc) + '\t')
                            #         file1.write('\n')
                            #     file1.close() 

                            #     if t == 4:
                            #         # Save gt
                            #         file1 = open(dis_file_path, "a")  # append mode 
                            #         # print(t, ii, seg_id)
                            #         file1.write('100' + '\t' + str(ii)  + '\n')
                            #         for rr in target_gt_dis:
                            #             for cc in rr:
                            #                 file1.write(str(cc) + '\t')
                            #             file1.write('\n')
                            #         file1.close() 

                            normalized = (final_dis - np.mean(final_dis))/np.std(final_dis)
                            # nss
                            scanpath, total_points = get_scanpath_from_center(future_second_fov, tile_map)
                            nss_score = np.sum(normalized * scanpath)/total_points
                            # print(nss_score)

                            losses[ii].append(cross_entropy)
                            nsses[ii].append(nss_score)
                            # print(losses[ii])

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

                        
                        if t == 1:
                            output = target_model.predict([target_prev_dis, the_last_seg_dis, pred_interval])
                        else:
                            # prepare other mean
                            future_other_ave = [np.zeros((Config.n_pitch, Config.n_yaw)) for _ in range(5)]
                            future_other_var = [np.zeros((Config.n_pitch, Config.n_yaw)) for _ in range(5)]
                            distirbutions = [[] for _ in range(5)]
                            for jjj in range(intervals):
                                future_seg_id = seg_id + jjj + 1
                                for jj in range(len(other_users)):

                                    other_fovs = video_fovs[other_users[jj]][1:]
                                    other_future_seg_fov = other_fovs[future_seg_id][1:]
                                    other_future_distributions = get_distribution_from_center(other_future_seg_fov, tile_map)
                                    distirbutions[jjj] += [other_future_distributions]
                                    # future_other_ave[jjj] += other_future_distributions
                            distirbutions = np.array(distirbutions)
                            for jjj in range(intervals):
                                future_other_ave[jjj] = np.mean(distirbutions[jjj], axis = 0)
                                future_other_var[jjj] = np.std(distirbutions[jjj], axis = 0)

                            # future_other_ave = np.array(future_other_ave)
                            num_other = np.ones((intervals,1))*(num_others/31)
                            num_other = np.expand_dims(num_other, axis=0)

                            future_other_ave = np.expand_dims(future_other_ave, axis=0)
                            future_other_var = np.expand_dims(future_other_ave, axis=0)
                            # other_std = np.expand_dims(other_std, axis=0)
                            future_other_ave = np.expand_dims(future_other_ave, axis=4)
                            future_other_var = np.expand_dims(future_other_ave, axis=4)

                            output = model.predict([target_prev_dis, the_last_seg_dis, future_other_ave, future_other_var, num_other, pred_interval_new])

                        output = output[0].reshape((intervals, Config.n_pitch, Config.n_yaw))
                        for ii in range(intervals):
                            absolute_seg_id = seg_id + ii + 1
                            future_second_fov = user_fov[absolute_seg_id][1:]
                            target_gt_dis = get_distribution_from_center(future_second_fov, tile_map)
                            if t == 2:
                                cross_entropy = utils.tile_cross_entropy(target_gt_dis, output[ii])
                            else:
                                cross_entropy = utils.tile_cross_entropy(target_gt_dis, output[ii])

                            final_dis = output[ii]
                            normalized = (final_dis - np.mean(final_dis))/np.std(final_dis)
                            # nss
                            scanpath, total_points = get_scanpath_from_center(future_second_fov, tile_map)
                            nss_score = np.sum(normalized * scanpath)/total_points
                            # print(nss_score)

                            losses[ii].append(cross_entropy)
                            nsses[ii].append(nss_score)

                            # if seg_id in random_step_id and vid == target_vid and i == target_user:
                            #     # find the path
                            #     dis_file_path = './fov_prediction/pred_dis/' + str(seg_id) + '.txt'
                            #     file1 = open(dis_file_path, "a")  # append mode 
                            #     file1.write(str(t) + '\t' + str(ii) +  '\n') 
                            #     for rr in final_dis:
                            #         for cc in rr:
                            #             file1.write( str(cc) + '\t')
                            #         file1.write('\n')
                            #     file1.close()  
            # print(losses)
            all_loss[t] += [losses]
            all_nss[t] += [nsses]
    # Collect data
    new_format = {}
    for key, value in all_loss.items():
        new_format[key] = [[] for _ in range(intervals)]
        # key is type of prediction, value is all data
        # print('type is:', key, value)
        curr_type_data = value
        for vid in range(len(value)):
            curr_video_data = value[vid]
            # print('vid: ', vid, curr_video_data)
            # there are five in each
            for interval in range(intervals):
                type_video_interval_data = curr_video_data[interval]
                new_format[key][interval] += type_video_interval_data

    # new_format_nss = {}
    # for key, value in all_nss.items():
    #     new_format_nss[key] = [[] for _ in range(5)]
    #     # key is type of prediction, value is all data
    #     # print('type is:', key, value)
    #     # curr_type_data = value
    #     for vid in range(len(value)):
    #         curr_video_data = value[vid]
    #         # print('vid: ', vid, curr_video_data)
    #         # there are five in each
    #         for interval in range(5):
    #             type_video_interval_data = curr_video_data[interval]
    #             new_format_nss[key][interval] += type_video_interval_data



    if not os.path.isdir('./fov_prediction/'):
        os.makedirs('./fov_prediction/')
    log_path = './fov_prediction/accuracy_first' + str(range_id) + '.txt'

    accuracy_write = open(log_path, 'w')

    for key, value in new_format.items():
        curr_bars = []
        print("type: ", key)
        for interval in range(len(value)):
            print('INterval ', interval)
            print("kl: ", np.mean(value[interval]))

            accuracy_write.write(str(np.mean(value[interval])) + ' ')
        accuracy_write.write('\n')

    # nss_log_path = './fov_prediction/nss_range' + str(range_id) + '.txt'

    # nss_write = open(nss_log_path, 'w')

    # for key, value in new_format_nss.items():
    #     curr_bars = []
    #     print("type: ", key)
    #     for interval in range(len(value)):
    #         print('INterval ', interval)
    #         print("nss: ", np.mean(value[interval]))

    #         nss_write.write(str(np.mean(value[interval])) + ' ')
    #     nss_write.write('\n')


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

if __name__ == '__main__':
    main()