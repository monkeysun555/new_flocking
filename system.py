import os
import numpy as np
import server 
import client
import utils
import queue as Q
from config import Config
from sklearn.cluster import KMeans
from load_models import *
from scipy.io import loadmat

def main():
    np.random.seed(Config.randomSeed)
    if not os.path.isdir(Config.figure_path):
        os.makedirs(Config.figure_path)
    if not os.path.isdir(Config.download_seq_path):
        os.makedirs(Config.download_seq_path)
    if not os.path.isdir(Config.info_data_path):
        os.makedirs(Config.info_data_path)
    # if Config.enable_cache:
    #     if not os.path.isdir(Config.represent_file_path):
    #         os.makedirs(Config.represent_file_path)

    ###########3 This is for feature map input conv_heatmap, v6 ##########################
    pred_interval = np.ones((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))
    for i in range(Config.target_user_pred_len):
        pred_interval[i] *= (i+1)/Config.target_user_pred_len/Config.n_pitch/Config.n_yaw
    pred_interval = np.expand_dims(pred_interval, axis=0)
    ##############################################################################
        
    ############# THis is for function temporal decay, v7 ######################
    pred_interval_new = np.ones((Config.target_user_pred_len,1))
    for i in range(Config.target_user_pred_len):
        pred_interval_new[i] *= (i+1)/Config.target_user_pred_len
    pred_interval_new = np.expand_dims(pred_interval_new, axis=0)
    ##############################################################################


    # Load tile map
    tile_map = loadmat(Config.tile_map_dir_new)['map']
    q_r_weight = loadmat(Config.qr_map_dir)
    co_a = utils.expand_yaw(q_r_weight['a'][0], Config.n_yaw)
    co_b = utils.expand_yaw(q_r_weight['b'][0], Config.n_yaw)
    co_w = utils.expand_yaw(q_r_weight['weight'][0], Config.n_yaw)
    # print(co_a[8][0])
    # print(co_b[8][0])
    # print([r[0] for r in co_w])
    # return
    # Get traces for simulation
    video_fovs, v_length = utils.load_fovs_for_video()        # (48, *) fovs for one video
    # print(v_length)
    # return
    # return
    # Manully change v length for TMM
    v_length = min(Config.v_length, v_length)

    if Config.USE_5G:
        time_traces, bandwidth_traces = utils.load_bw_traces()
    else:
        time_traces, bandwidth_traces = utils.load_bw_traces_new_5g()

    ## to show rsd distribution of first 48 traces for enhanced
    # utils.show_statictic_enhanced(bandwidth_traces[:Config.num_users])
    #############   
    # Create server
    video_server = server.Server(tile_map)
    server_time, server_seg_idx = video_server.get_current_info()
    current_time = server_time
    # Create usrs
    next_timestamp = []
    user_group = []
    groups_print = {}

    ## Create general fov prediction model (used by all users)
    if Config.new_prediction_version == 6:
        model = load_keras_model()
        target_model = load_keras_target_model()
        print("load model successfully!")

    elif Config.new_prediction_version == 7 or Config.new_prediction_version == 8 or Config.new_prediction_version == 11 or Config.new_prediction_version == 12:
        model = load_decay_model_new()
        if Config.new_prediction_version == 8:
            # target_model = load_keras_target_model('./keras_models/non_heatmap/kk.h5')
            target_model = load_keras_target_model()
        else:
            target_model = load_keras_target_model()
        print("load model successfully!")
    # return 

    for i in range(Config.num_users):
        user_fov = video_fovs[i]  
        print('user id', user_fov[0])         
        time_trace = time_traces[i]               # Time and bw traces use the same idx, mmsys use i+10
        bandwidth_trace = bandwidth_traces[i]     # Time and bw traces use the same idx

        # Will cause massive overhead work
        # Disabled
        # additional_latency = np.random.randint(Config.enhanced_extra_latency)         # Additional latency after servert starts encoding
        ###########################
        if Config.latency_optimization:
            if Config.USE_5G:
                l_group_idx, init_latency, buffer_upper = utils.get_group_idx_optimized_enhanced(bandwidth_trace)    # For enhanced, use small period of time to assign latency group
            else:
                l_group_idx, init_latency, buffer_upper = utils.get_group_idx_optimized_enhanced_new_5g(i)    # For enhanced, use small period of time to assign latency group
            print(i,l_group_idx, init_latency, buffer_upper)
            # return
            if l_group_idx in groups_print:
                groups_print[l_group_idx]+=1
            else:
                groups_print[l_group_idx]=1
            # l_group_idx, init_latency, buffer_upper = utils.get_group_idx_optimized(bandwidth_trace)    # For enhanced, use small period of time to assign latency group
            user = client.User(i, l_group_idx, init_latency, user_fov, \
                    time_trace, bandwidth_trace, server_seg_idx, v_length, current_time, matrix = np.array([co_a, co_b, co_w]), tile_map = tile_map, buffer_upper_bound=buffer_upper)
        else:
            # l_group_idx, init_latency = utils.get_group_idx()
            # Latency gropu is random assigned
            l_group_idx, init_latency, buffer_upper = utils.get_group_idx_equal(i)
            user = client.User(i, l_group_idx, init_latency, user_fov, \
                    time_trace, bandwidth_trace, server_seg_idx, v_length, current_time, matrix = np.array([co_a, co_b, co_w]), tile_map = tile_map, buffer_upper_bound=buffer_upper)
        user_group.append(user)
    print(groups_print)
    # Initial fov using initial latencies, push fov to server
    for u_id in range(len(user_group)):
        user = user_group[u_id]
        fov_info = user.generate_fov_info()
        # To be modified, push to server
        if len(fov_info):
            video_server.collect_fov_info(user.get_id(), fov_info, user.get_next_req_seg_idx(), user.get_playing_time(), initial=True)

    # if Config.enable_cache:
    #     video_server.find_user_tiers()
    #     repre_id, not_repre_id, global_lowest_id = video_server.get_represent()
    #     represent_time = current_time
    #     represent_file_path = Config.represent_file_path + 'usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     repre_file = open(represent_file_path, 'w')
    #     repre_file.write(str(represent_time) + ' ' + str(repre_id[0])+ ' ' + str(repre_id[1]) + ' ' + str(repre_id[2]) + ' ' + str(repre_id[3]))
    #     repre_file.write('\n')
    #     repre_weight = dict()
    #     for beta in range(Config.num_users):
    #         repre_weight[beta] = 0.0

    #     # Not repre list, lowest of each group
    #     not_represent_file_path = Config.represent_file_path + 'not_usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     not_repre_file = open(not_represent_file_path, 'w')
    #     not_repre_file.write(str(represent_time) + ' ' + str(not_repre_id[0])+ ' ' + str(not_repre_id[1]) + ' ' + str(not_repre_id[2]) + ' ' + str(not_repre_id[3]))
    #     not_repre_file.write('\n')

    #     # Lowest several global
    #     global_not_represent_file_path = Config.represent_file_path + 'global_not_usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     global_not_repre_file = open(global_not_represent_file_path, 'w')
    #     global_not_repre_file.write(str(represent_time))
    #     for global_lowest in global_lowest_id:
    #         global_not_repre_file.write(' ' + str(global_lowest)) 
    #     global_not_repre_file.write('\n')

    #     # Global not weight
    #     global_not_w_represent_file_path = Config.represent_file_path + 'global_not_w_usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     global_not_w_repre_file = open(global_not_w_represent_file_path, 'w')
    #     global_not_w_repre_file.write(str(represent_time))
    #     for global_lowest in global_lowest_id:
    #         global_not_w_repre_file.write(' ' + str(global_lowest)) 
    #     global_not_w_repre_file.write('\n')

    # Initial next timestap heap for users
    event_q = Q.PriorityQueue()

    for u_id in range(len(user_group)):
        next_time, events = user_group[u_id].sim_fetching(server_time)
        event_q.put((next_time, (u_id, events, 0)))
        if Config.debug:
            print("Next time is: ", next_time)
            print(events)

    # Finish initial, get next updated user and then move system state
    # if Config.enable_cache:
    #     download_seq = []
    #     current_cache_file_idx = 0

    while not event_q.empty():
        next_user_info = event_q.get()
        next_time = next_user_info[0]
        u_id = next_user_info[1][0]
        u_event = next_user_info[1][1]
        u_action = next_user_info[1][2]
        user = user_group[u_id]
        # Do real system evolution
        if u_action == 0:
            user_download_seg_idx = user.get_download_seg_idx()
            # Download next segment
            seg_info, download_time = user.fetching()
            tile_idxes, tile_rates = user.get_download_tile_info()
            user.record_downloaded_tiles([user_download_seg_idx, seg_info, tile_rates])
            user.update_bw_his(seg_info)
            # if Config.enable_cache:
            #     # Insert tiles info to cache
            #     offset_time = user.get_prefetching_time()
            #     new_off = offset_time
            #     offset_time += seg_info[0]
            #     for tile in seg_info[1:]:
            #         if tile[1] == tile[2]:
            #             offset_time += tile[3]
            #             # Tile id is (seg_idx, pitch_idx, and yaw_idx)
            #             rate = tile_rates[tile_idxes.index(tile[0])]
            #             tile_id = (user_download_seg_idx, tile[0][0], rate)
            #             download_seq.append((offset_time, tile_id, u_id))
            #             if len(download_seq) > 5000:
            #                 np.savetxt(Config.download_seq_path + 'usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '_index' + str(current_cache_file_idx)+ '.txt', download_seq, fmt='%s')
            #                 download_seq = []
            #                 current_cache_file_idx += 1
            if Config.debug:
                print("Playing time after fetching: ", user.get_playing_time())
                print("Buffer len: ", user.buffer_len)
                print("Seg info: ", seg_info)
                print("Sim download time: ", u_event[-1][1] - u_event[0][0])
                # print(sum([seg[3] for seg in seg_info[1:]]), seg_info[0])
                print("Real download time: ", download_time)
            # assert np.round(u_event[-1][1] - u_event[0][0]) == np.round(sum([seg[3] for seg in seg_info[1:]]) + seg_info[0])
        elif u_action == 1:
            # Wait
            wait_time = next_user_info[1][3]
            user.wait(wait_time)
        if Config.debug and u_id == 0:
            print("u 0 next time: ", next_time)
            print("gap is: ", next_time-current_time)
        video_server.update(next_time-current_time)
        current_time = next_time

        # # ##############################################################
        # # Get represent from server calculate distance, seems not work well
        # # if Config.enable_cache:
        # #     if current_time - represent_time >= Config.represent_update_interval:
        # #         represent_time = current_time
        # #         repre_id = video_server.get_represent()
        # #         repre_file.write(str(represent_time) + ' ' + str(repre_id))
        # #         repre_file.write('\n')
        # # ##############################################################
        user.udpate_prefetching_time(current_time)
        # user.adjust_playing_time()

        # Check whether buffer exceed or server wait
        if Config.debug:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        server_time, server_seg_idx = video_server.get_current_info()
        player_buffer_len = user.get_buffer_length()
        # print("Player buff: ", player_buffer_len, user.get_buffer_upper_bound())
        # print(user.playing_time, user.next_req_seg_idx)
        user_req_seg_idx = user.get_next_req_seg_idx()
        buffer_wait_time = max(0.0, player_buffer_len - user.get_buffer_upper_bound())
        # print("Wait time is:", buffer_wait_time)
        server_wait_time = 0.0
        # assert user_req_seg_idx <= server_seg_idx
        if user_req_seg_idx == server_seg_idx:
            server_wait_time = Config.seg_duration - server_time%Config.seg_duration
        wait_time = max(buffer_wait_time, server_wait_time)
        if wait_time == 0:
            # Upload fov to server
            fov_info = user.generate_fov_info()
            if len(fov_info):
                video_server.collect_fov_info(u_id, fov_info, user.get_next_req_seg_idx(), user.get_playing_time())
        
        # Do prediction on this user using all infomation
        if wait_time > 0.0:
            next_time, events = user.sim_wait(server_time, wait_time)
            event_q.put((next_time, (u_id, events, 1, wait_time)))
            if Config.debug:
                print("Next time is (wait): ", next_time)
                print(events)
        else:
            if user.check_ending():
                # No more fov trace
                continue
            
            # If next step is to download, predict bw and fov
            ####################################################################
            predicted_bw = user.predict_bw()
            pred_type = None
            if Config.new_prediction_version == 6 or Config.new_prediction_version == 7 or Config.new_prediction_version == 8 or Config.new_prediction_version == 11 or Config.new_prediction_version == 12:

                # First of all, check target seg/dis
                curr_playing_seg, predicted_seg_idx = user.tmm_check_target_display()
                if Config.new_prediction_version == 8:
                    output_offset = min(max(predicted_seg_idx - curr_playing_seg, 0), 4)
                else:
                    output_offset = min(max(predicted_seg_idx - curr_playing_seg - 1, 0), 4)

                
                # output_offset = min(predicted_seg_idx - curr_playing_seg, Config.target_user_pred_len-1)
                
                # print(predicted_seg_idx, curr_playing_seg)
                # Check available others, if none, do self prediction
                target_prev_dis = user.tmm_get_target_fov_info()
                the_last_seg_dis = np.expand_dims(target_prev_dis[-1], axis=0)
                target_prev_dis = np.expand_dims(target_prev_dis, axis=0)
                the_last_seg_dis = np.expand_dims(the_last_seg_dis, axis=0)
                
                other_mean, other_std, other_numbers = video_server.get_user_fovs_tmm(curr_playing_seg, predicted_seg_idx)
                # Then get target users
                # prepared_info, predicted_seg_idx, display_segs, gap_in_s = user.tmm_get_target_fov_info()

                if Config.new_prediction_version == 8 or len(other_mean) <= 1 or output_offset <= 1 :
                    pred_type = 1
                    # Do target self prediction
                    if user.get_id() == 5:
                        print('zero for user 5')
                    output = target_model.predict([target_prev_dis, the_last_seg_dis, pred_interval])
                else:
                    pred_type = 2
                    if Config.new_prediction_version == 6:
                        # 2nd Use model to predict
                        # prepare model input
                        num_other = np.ones((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))*other_numbers/Config.num_users/Config.n_pitch/Config.n_yaw
                        # num_other = np.ones((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))*len(other_mean)/48
                        num_other = np.expand_dims(num_other, axis=0)
                        other_mean = np.expand_dims(other_mean, axis=0)
                        other_std = np.expand_dims(other_std, axis=0)
                        if Config.debug:
                            print('Output from model')
                            print(output.shape)
                            print(target_prev_dis.shape)
                            print(the_last_seg_dis.shape)
                            print(other_mean.shape)
                            print(other_std.shape)

                        output = model.predict([target_prev_dis, the_last_seg_dis, other_mean, other_std, num_other, pred_interval])

                    else:
                        num_other = np.ones((Config.target_user_pred_len,1))*(other_numbers/Config.num_users)
                        num_other = np.expand_dims(num_other, axis=0)

                        other_mean = np.expand_dims(other_mean, axis=0)
                        other_std = np.expand_dims(other_std, axis=0)
                        output = model.predict([target_prev_dis, the_last_seg_dis, other_mean, other_std, num_other, pred_interval_new])
                    # 3rd The do rate allocation
                # print(output.shape)
                output = output[0].reshape((Config.target_user_pred_len, Config.n_pitch, Config.n_yaw))
                # print(output)
                # Get the distribution for the segment
                pred_seg_distribution = output[output_offset]
                # print(pred_seg_distribution)
                # print(np.sum(pred_seg_distribution))
                # print(pred_seg_distribution)
                user.record_pred_distribution(pred_seg_distribution, output_offset, pred_type)
                user.rate_allocation(pred_seg_distribution, predicted_bw)

            else:
                prepared_info, predicted_seg_idx, display_segs, predicted_center, gap_in_s = user.predict_fov()

                ######################## for mmsys #############################
                # neighbor_fovs, neighbor_dis_maps, nei_target_traces = video_server.get_user_fovs(display_segs, predicted_seg_idx)
                # betas = user.choose_tiles(predicted_seg_idx, predicted_bw, prepared_info, predicted_center, neighbor_fovs, neighbor_dis_maps, gap_in_s, nei_target_traces) 
                ################################################################
                
                # Below is for enhanced
                neighbor_fovs, neighbor_dis_maps, nei_target_traces, seg_ave_from_server = \
                                video_server.get_user_fovs_enhanced(display_segs, predicted_seg_idx)
                user.choose_tiles(predicted_seg_idx, predicted_bw, prepared_info, predicted_center, \
                            neighbor_fovs, neighbor_dis_maps, gap_in_s, nei_target_traces, seg_ave_from_server) 

            # if Config.enable_cache:
            #     for beta in betas:
            #         repre_weight[beta[0]] += beta[1]
            #     if current_time - represent_time >= Config.represent_update_interval:
            #         # First of all, cluster users based on latency
            #         represent_time = current_time
            #         tmp_repre_list = []
            #         time_info_list = [0.0]*Config.num_users
            #         for user_id in range(Config.num_users):
            #             tmp_user = user_group[user_id]
            #             user_playing_time = tmp_user.get_playing_time()
            #             time_info_list[user_id] = (user_id, user_playing_time)
            #         # Do clustering
            #         time_info_list.sort(key=lambda x:x[1], reverse=True)

            #         # # Method 1: Select the one with highest weight from each group 
            #         # for group_i in range(4):    # Four groups
            #         #     group_users_id = [u[0] for u in time_info_list[group_i*int(Config.num_users/4):(group_i+1)*int(Config.num_users/4)]]
            #         #     curr_group_weight = [(g_user_id, repre_weight[g_user_id]) for g_user_id in group_users_id]
            #         #     curr_group_repre = max(curr_group_weight, key=lambda x:x[1])[0]
            #         #     tmp_repre_list.append(curr_group_repre)

            #         # Method 2: Select the one with shortest latecy in each group
            #         diff = [(u_time_idx, time_info_list[u_time_idx][1] - time_info_list[u_time_idx-1][1])  for u_time_idx in range(1, len(time_info_list))]
            #         diff.sort(key=lambda x:x[1])
            #         for break_id in diff[:3]:
            #             tmp_repre_list.append(break_id[0])
            #         tmp_repre_list.append(time_info_list[0][0])
            #         assert len(tmp_repre_list) == 4
            #         repre_file.write(str(represent_time) + ' ' + str(tmp_repre_list[0])+ ' ' + str(tmp_repre_list[1]) + ' ' + str(tmp_repre_list[2]) + ' ' + str(tmp_repre_list[3]))
            #         repre_file.write('\n')

            #         # Find lowest users
            #         not_tmp_repre_list = []
            #         for break_id in diff[:3]:
            #             not_tmp_repre_list.append(break_id[0]-1)
            #         not_tmp_repre_list.append(time_info_list[-1][0])
            #         not_repre_file.write(str(represent_time) + ' ' + str(not_tmp_repre_list[0])+ ' ' + str(not_tmp_repre_list[1]) + ' ' + str(not_tmp_repre_list[2]) + ' ' + str(not_tmp_repre_list[3]))
            #         not_repre_file.write('\n')

            #         # Global lowest 10 users
            #         global_not_repre_list = [x[0] for x in time_info_list[int(-Config.num_users/6):]]
            #         global_not_repre_file.write(str(represent_time))
            #         for global_not_repre in global_not_repre_list:
            #             global_not_repre_file.write(' ' + str(global_not_repre))
            #         global_not_repre_file.write('\n')

            #         # Global lowest 10 users
            #         global_not_w_repre_list = [x[0] for x in time_info_list[int(-Config.num_users/4):]]
            #         curr_w_group_weight = [(g_user_id, repre_weight[g_user_id]) for g_user_id in global_not_w_repre_list]
            #         curr_w_group_weight.sort(key=lambda x:x[1])

            #         global_not_w_repre = [x[0] for x in curr_w_group_weight[:int(Config.num_users/6)]]
            #         global_not_w_repre_file.write(str(represent_time))
            #         for global_not_repre in global_not_w_repre:
            #             global_not_w_repre_file.write(' ' + str(global_not_repre))
            #         global_not_w_repre_file.write('\n')
                    
            #         for key, value in repre_weight.items():
            #             repre_weight[key] = 0.0
            ####################################################################
            ####################################################################
            next_time, events = user.sim_fetching(server_time)
            event_q.put((next_time, (u_id, events, 0)))
            if Config.debug:
                print("Next time is (fetching): ", next_time)
                print(events)
        # Put back to pq

    # Record all mapping/trace/userid
    mapping_path = Config.info_data_path + 'Mapping_usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    log_file = open(mapping_path, 'w')

    # Evaluate performance
    for user in user_group:
        print('Saving data user: ', user.get_id())
        rates, freeze, cross_entropy, overlap_ratio = user.evaluate_performance()
        time_trace, bw_trace = user.get_sim_bw_trace()
        # utils.plot_metrics(user.get_id(), time_trace, bw_trace, rates, freeze)
        log_file.write('User: ' + str(user.get_id()) + ' FoV ID: ' + str(user.fov_trace_id) + ' BW Trace: ' + str(np.mean(user.bw_trace)) + ' Latency group: ' + str(user.l_group_idx) )
        log_file.write('\n')
        log_file.write(' # Segs: ' + str(len(rates)) + ' Ave rates: ' + str(np.mean([rate[1] for rate in rates])))
        log_file.write(' Total freezing: ' + str(np.sum([f[1] for f in freeze])))
        log_file.write(' Average cross_entropy: ' + str(np.mean([c_e[1] for c_e in cross_entropy])) + ' ' + str(np.mean([c_e[1] for c_e in overlap_ratio])))
        log_file.write('\n')

        # log_file.write
        
    # if Config.enable_cache:
    #     np.savetxt(Config.download_seq_path + 'usrs' + str(Config.num_users) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '_index' + str(current_cache_file_idx)+ '.txt', download_seq, fmt='%s')
    # #     edge_cache = utils.edge_cache()
    # #     edge_cache.do_caching(download_seq)

if __name__ == '__main__':
    main()
