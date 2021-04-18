import os
import numpy as np
import matplotlib.pyplot as plt
from config import Config
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from utils import *

# Note:
# 1. ratio between I and P is 10~15

class Server(object):
    def __init__(self, tile_map):
        np.random.seed(Config.randomSeed)
        # Configuration
        self.seg_duration = Config.seg_duration
        self.n_yaw = Config.n_yaw
        self.n_pitch = Config.n_pitch

        # Server encoding state
        self.video_rates = Config.bitrate
        self.encoding_time = Config.initial_latencies[Config.latency_control_version]*Config.seg_duration + np.random.random()*Config.seg_duration
        self.current_seg_idx = 0
        self.encoding_buffer = []
        self.update_encoding_buffer(0.0, self.encoding_time)
        # Server connection state
        self.n_clients = Config.num_users
        # self.client_info = []

        # Server FoV collection
        self.client_fovs = dict()
        self.req_seg_idx_list = dict()
        self.playing_time_list = dict()
        self.fov_count = 0
        self.last_seg_idx = 0
        # for DBSCAN clustering
        self.scaler = StandardScaler()
        if Config.transform_sim == 1:
            self.dbscan = DBSCAN(eps=Config.dbscan_eps, metric=transform_similarity)
        elif Config.transform_sim == 2:
            self.dbscan = DBSCAN(eps=Config.dbscan_eps, metric=sphere_distance_new)
        elif Config.transform_sim == 0:
            self.dbscan = DBSCAN(eps=Config.dbscan_eps, metric=similarity)

        self.tile_map = tile_map
        self.distribution_map = dict()  #key: user_id, value [[seg_id, xx, xx], [seg_id, xx,xx]]
        self.seg_clustered_distribution = dict()            # key: seg_id, value: distribution

        self.highest_user_idx = None
        self.lowest_user_idx = None
        self.global_lowest_idx = None

    def update(self, downloadig_time):
        # update time and encoding buffer
        # Has nothing to do with sync, migrate to player side
        # sync = 0  # sync play
        # missing_count = 0
        # new_heading_time = 0.0
        pre_time = self.encoding_time
        self.encoding_time += downloadig_time
        self.update_encoding_buffer(pre_time, self.encoding_time)
        # Generate new delivery for next
        # self.generate_next_delivery()

        # # Check delay threshold
        # # A: Triggered by server sice, not reasonable
        # if len(self.chunks) > 1:
        #   if self.time - playing_time > self.delay_tol:
        #       new_heading_time, missing_count = self.sync_encoding_buffer()
        #       sync = 1

        # # B: Receive time_out from client, and then resync
        # if time_out:
          # assert len(self.chunks) > 1
        #   new_heading_time, missing_count = self.sync_encoding_buffer()
        #   sync = 1
        # return sync, new_heading_time, missing_count
        print("Server current time: ", self.encoding_time)
        return

    def update_encoding_buffer(self, start_time, end_time):
        temp_time = start_time
        while True:
            next_time = (int(temp_time/self.seg_duration) + 1) * self.seg_duration
            if next_time > end_time:
                break
            # Generate chunks and insert to encoding buffer
            temp_time = next_time
            # If it is the first chunk in a seg
            seg_tiles_size = self.generate_chunk_size()
            self.encoding_buffer.append(seg_tiles_size) 
            self.current_seg_idx += 1 

    def generate_chunk_size(self):
        if Config.encoding_allocation_version == 0:
            # Uniform distribution
            ratio_for_each = 1.0/(self.n_yaw*self.n_pitch)
            seg_tiles_size = [self.current_seg_idx]
            for i in range(len(self.video_rates)):
                vidoe_rate = self.video_rates[i]
                tiles_size = []
                for y_idx in range(self.n_yaw):
                    for p_idx in range(self.n_pitch):
                        tiles_size.append(np.random.normal(ratio_for_each, 0.01*ratio_for_each)*self.seg_duration*vidoe_rate)
                seg_tiles_size.append(tiles_size)
            return seg_tiles_size

    def get_current_info(self):
        return self.encoding_time, self.current_seg_idx

    def find_user_tiers(self):
        # Find user tiers for caching
        curr_playing_time_list = []
        tmp_repre_list = []
        not_tmp_repre_list = []
        for key, value in self.playing_time_list.items():
            curr_playing_time_list.append((key, value))
        curr_playing_time_list.sort(key=lambda x:x[1], reverse = True)
        n_first_tier = int(Config.num_users/4)
        #################### Find single repre ####################
        # first_tier_user = [x[0] for x in curr_playing_time_list[:n_first_tier]]
        # rate_for_f_users = [0.0] * n_first_tier
        # for i in range(n_first_tier, len(curr_playing_time_list)):
        #     uid = curr_playing_time_list[i][0]
        #     current_fov_trace = [(point[1][0], point[1][1]) for point in self.client_fovs[uid][0][1]]
        #     current_seg_idx = self.client_fovs[uid][0][0]
        #     distance_trace = []
        #     for first_tier_idx in range(len(first_tier_user)):
        #         f_user_id = first_tier_user[first_tier_idx]
        #         f_user_trace = [(point[1][0], point[1][1]) for point in self.client_fovs[f_user_id][0][1]]
        #         assert self.client_fovs[f_user_id][0][0] == current_seg_idx
        #         distance = calculate_curve_distance(current_fov_trace, f_user_trace)
        #         distance_trace.append((first_tier_idx, distance_trace)) # Using index of usr, not real uid
        #     closest_f_user = min(distance_trace, key=lambda x:x[1])[0]  # Get min index
        #     rate_for_f_users[closest_f_user] += 1.0

        # # Using last seg trace in self.client_fovs to calculate distance
        # highest_user_idx = rate_for_f_users.index(max(rate_for_f_users))
        # self.highest_user_idx = first_tier_user[highest_user_idx]
        ##########################################################################
        for tier_id in range(4):
            repre_id = curr_playing_time_list[tier_id*n_first_tier][0]
            not_repre_id = curr_playing_time_list[(tier_id+1)*n_first_tier-1][0]
            tmp_repre_list.append(repre_id)
            not_tmp_repre_list.append(not_repre_id)
        self.highest_user_idx = tmp_repre_list
        self.lowest_user_idx = not_tmp_repre_list
        self.global_lowest_idx = [x[0] for x in curr_playing_time_list[int(-Config.num_users/6):]]

    def get_represent(self):
        return self.highest_user_idx, self.lowest_user_idx, self.global_lowest_idx

    def delete_fov_table(self):
        for key, value in self.client_fovs.items():
            i = 0
            while i < len(value):
                # print(value[i][0])
                if np.floor(value[i][0]) < self.last_seg_idx:
                    i += 1
                else:
                    # assert np.floor(value[i][0]) == self.last_seg_idx
                    break
            del self.client_fovs[key][:i]

        for key, value in self.distribution_map.items():
            i = 0
            while i < len(value):
                # print(value[i][0])
                if np.floor(value[i][0]) < self.last_seg_idx:
                    i += 1
                else:
                    # assert np.floor(value[i][0]) == self.last_seg_idx
                    break
            del self.distribution_map[key][:i]

        if Config.fov_update_per_upload:
            pass
        # print('after delete')
        # print(last_seg_idx)
        # for i in range(len(self.client_fovs)):
        #     print(self.client_fovs[i])

    def update_saliency_map_per_upload(self, u_id, len_of_update_info):
        # u_id_len = len(self.client_fovs[u_id])
        # for seg_idx in reversed(range(len_of_update_info)):
        #     # For fov for each seg
        #     # Reversed direction
        #     data = [value[u_id_len-seg_idx-1] for key, value in self.client_fovs.items() if len(value) >= u_id_len - seg_idx]
        #     print(len(data))
        pass
    
    def update_saliency_map_per_interval(self):
        # Method based on the sampling
        # seg_i = 1
        # tmp_saliency_map = []
        # while True:
        #     data = [(key, value[seg_i-1]) for key, value in self.client_fovs.items() if len(value) >= seg_i]   # The if is checking whether fov info for the seg idx is empty
        #     curr_seg_saliency = []
        #     if len(data) >= Config.DBSCAN_tth:
        #         tmp_seg_idx = data[0][1][0]
        #         curr_seg_saliency.append(tmp_seg_idx)
        #         for r_id in range(Config.num_fov_per_seg):
        #             tmp_u_id = [data[i][0] for i in range(len(data))]
        #             tmp_data = [data[i][1][1][r_id] for i in range(len(data))]
        #             # scaled_data = self.scaler.fit_transform(tmp_data)
        #             scaled_data = tmp_data
        #             clusters = self.dbscan.fit_predict(scaled_data)
        #             curr_ckp_clustered_data = np.array((tmp_u_id, tmp_data, clusters)).T.tolist()
        #             if Config.show_cluster:
        #                 plt.scatter([p[0] for p in tmp_data], [p[1] for p in tmp_data], c=clusters, cmap="plasma")
        #                 plt.show()
        #                 input()
        #             curr_seg_saliency.append(curr_ckp_clustered_data)
        #         seg_i += 1
        #         tmp_saliency_map.append(curr_seg_saliency)
        #     else:
        #         # There is no more fov with enough records
        #         # Update saliency map and return
        #         self.saliency_map_list = tmp_saliency_map
        #         return
        
        ## Build up saliency map for a user using fov (for all frames) for one video segment
        seg_i = 1
        while True:
            # Check for all users
            tmp_saliency_maps = []
            user_datas = [(key, value[seg_i-1]) for key, value in self.client_fovs.items() if len(value) >= seg_i]   # The if is checking whether fov info for the seg idx is empty
            # Build up saliency map from centers of frames
            if len(user_datas):
                # user_datas: [uid, [seg_id, [time, (yaw, pitch, roll)], [time, (yaw, pitch, roll)],...[time, ()]]]
                for user_info in user_datas:
                    # For different users
                    u_id = user_info[0]
                    real_seg_idx = user_info[1][0]
                    frames_info = user_info[1][1]
                    # print("frames info: ", frames_info)
                    distribution = self.get_distribution_from_center(frames_info)

                    if u_id in self.distribution_map.keys():
                        # print(real_seg_idx, self.distribution_map[u_id][-1][0])
                        if len(self.distribution_map[u_id]) and real_seg_idx <= self.distribution_map[u_id][-1][0]:
                            continue
                        self.distribution_map[u_id].append([real_seg_idx, distribution, 1])
                    else:
                        self.distribution_map[u_id] = [[real_seg_idx, distribution, 1]]
                seg_i += 1
            else:
                return

    def update_seg_ave_distribution_dbscan(self):
        seg_i = 1
        while True:
            # Check for all users
            tmp_saliency_maps = []
            user_datas = [(key, value[seg_i-1]) for key, value in self.client_fovs.items() if len(value) >= seg_i]   # The if is checking whether fov info for the seg idx is empty
            # Build up saliency map from centers of frames
            if len(user_datas):
                real_seg_idx = user_datas[0][1][0]
                curr_len = len(user_datas) 
                seg_data = []
                for user_info in user_datas:
                    # For different users
                    u_id = user_info[0]
                    # real_seg_idx = user_info[1][0]
                    frames_info = user_info[1][1]
                    data_position = [(x[1][0], x[1][1]) for x in frames_info]
                    seg_data.append(data_position)
                    # print(data_position)

                m_l = min([len(x) for x in seg_data])
                seg_data = [x[:m_l] for x in seg_data]
                for k in range(1,len(seg_data)):
                    assert len(seg_data[k]) == len(seg_data[k-1])

                frames_dis = []
                filterd_users = []
                for f in range(len(seg_data[0])):
                    frame_data = [u_data[f] for u_data in seg_data]
                    if curr_len >= Config.dbscan_ave_tth:
                        clusters = self.dbscan.fit_predict(frame_data)
                        # curr_ckp_clustered_data = np.array((tmp_u_id, tmp_data, clusters)).T.tolist()
                        # if Config.show_cluster:
                        #     plt.scatter([p[0] for p in tmp_data], [p[1] for p in tmp_data], c=clusters, cmap="plasma")
                        #     plt.show()
                        #     input()
                        # for each frame, how the positions are clustered
                        filtered_data = []
                        for g in range(len(clusters)):
                            if clusters[g] >= 0:
                                filtered_data.append(frame_data[g])
                        # Generate frame filtered distribution
                    else:
                        filtered_data = frame_data
                        # To do
                    frame_dis = self.get_distribution_from_center_of_frame(filtered_data)

                    if np.round(np.sum(frame_dis), 1) == 1.0:
                        frames_dis.append(frame_dis)
                        filterd_users.append(len(filtered_data))
                        # print(1)
                    # else:
                        # print(np.round(np.sum(frame_dis), 1))
                        # print('0')

                # Get segemnt ave
                # print(np.array(frames_dis).shape)

                if len(frames_dis):
                    seg_dis = np.mean(np.array(frames_dis), axis=0).tolist()
                    user_weight = np.mean(filterd_users)
                    # print(np.round(np.sum(seg_dis), 1))
                    # print(seg_dis)
                    # assert np.round(np.sum(seg_dis), 1) == 1.0
                else:
                    seg_dis = np.zeros((Config.n_pitch, Config.n_yaw))
                    user_weight = 0
                # Store the seg_dis
                self.seg_clustered_distribution[real_seg_idx] = (user_weight, seg_dis)
                seg_i += 1
            else:
                return 


    def update_seg_ave_distribution(self):
        seg_i = 1
        while True:
            # Check for all users
            tmp_saliency_maps = []
            user_datas = [(key, value[seg_i-1]) for key, value in self.client_fovs.items() if len(value) >= seg_i]   # The if is checking whether fov info for the seg idx is empty
            # Build up saliency map from centers of frames
            if len(user_datas):
                if len(user_datas) >= Config.n_dis_tth:
                # user_datas: [uid, [seg_id, [time, (yaw, pitch, roll)], [time, (yaw, pitch, roll)],...[time, ()]]]
                    speeds = []
                    for user_info in user_datas:
                        # For different users
                        u_id = user_info[0]
                        real_seg_idx = user_info[1][0]
                        frames_info = user_info[1][1]
                        # Calculate accelerate
                        yaw_speed = np.diff([x[1][0] for x in frames_info])
                        pitch_speed = np.diff([x[1][1] for x in frames_info])
                        yaw_a = np.diff(yaw_speed)
                        pitch_a = np.diff(pitch_speed)
                        # Assign weight
                        speeds.append([u_id, np.mean([np.abs(s) for s in yaw_speed]), \
                                            np.mean([np.abs(s) for s in pitch_speed]), frames_info])

                    # print(speeds)
                    #Get Z score
                    y_speeds_mean = np.mean([s[1] for s in speeds])
                    y_speeds_std = np.std([s[1] for s in speeds])
                    p_speeds_mean = np.mean([s[2] for s in speeds])
                    p_speeds_std = np.std([s[2] for s in speeds])
                    for speed_idx in range(len(speeds)):
                        z = np.abs((speeds[speed_idx][1]-y_speeds_mean)/y_speeds_std)
                        if z >= 1:
                            # outlier
                            speeds[speed_idx].append(0)   # Add weight
                        else:
                            speeds[speed_idx].append(0.5*(1-z))
                    for speed_idx in range(len(speeds)):
                        frames_info = speeds[speed_idx][3]
                        weight = speeds[speed_idx][4]
                        # print("frames info: ", frames_info)
                        distribution = self.get_distribution_from_center(frames_info)

                        if u_id in self.distribution_map.keys():
                            self.distribution_map[u_id].append([real_seg_idx, distribution, weight])
                        else:
                            self.distribution_map[u_id] = [[real_seg_idx, distribution, weight]]
                else:
                    for user_info in user_datas:
                        # For different users
                        u_id = user_info[0]
                        real_seg_idx = user_info[1][0]
                        frames_info = user_info[1][1]
                        weight = 1
                        distribution = self.get_distribution_from_center(frames_info)
                        if u_id in self.distribution_map.keys():
                            self.distribution_map[u_id].append([real_seg_idx, distribution, weight])
                        else:
                            self.distribution_map[u_id] = [[real_seg_idx, distribution, weight]]
                seg_i += 1
            else:
                return

    def get_distribution_from_center_of_frame(self, frame_users_info):
        n_users = float(len(frame_users_info))
        frame_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
        if n_users:
            for u_info in frame_users_info:
                pitch_center = int(np.round(u_info[1]/np.pi*180)) + 90        # Index offset
                yaw_center = int(np.round(u_info[0]/np.pi*180)) + 180         # Index offset
                # print(pitch_center, yaw_center)
                tiles = self.tile_map[pitch_center][yaw_center]
                frame_distribution += np.array(tiles)/np.sum(tiles)     # Get total numbers for tiles for one frame, then normalized
            frame_distribution /= n_users
        return frame_distribution

    def get_distribution_from_center(self, frames_info):
        # This is for all frames within a segment
        n_frames = float(len(frames_info))
        frame_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
        for f_info in frames_info:
            pitch_center = int(np.round(f_info[1][1]/np.pi*180)) + 90        # Index offset
            yaw_center = int(np.round(f_info[1][0]/np.pi*180)) + 180         # Index offset
            # print(pitch_center, yaw_center)
            tiles = self.tile_map[pitch_center][yaw_center]
            frame_distribution += np.array(tiles)/np.sum(tiles)     # Get total numbers for tiles for one frame, then normalized
        frame_distribution /= n_frames
        return frame_distribution

    def find_tiles(fov_direction):
        yaw = fov_direction[0]
        pitch = fov_direction[1]

    def collect_fov_info(self, u_id, fov_info, req_seg_idx, playing_time, initial=False):
        # print(u_id, req_seg_idx)
        if u_id in self.client_fovs.keys():
            self.client_fovs[u_id].extend(fov_info)
        else:
            self.client_fovs[u_id] = fov_info
        self.req_seg_idx_list[u_id] = req_seg_idx
        self.playing_time_list[u_id] = playing_time
        if not initial:
            self.fov_count += 1
            if self.fov_count%Config.table_delete_interval == 0:
                # print('kllls')
                # for i in range(len(self.client_fovs)):
                #     print(len(self.client_fovs[i]))
                # print(self.req_seg_idx_list)
                # print(self.playing_time_list)
                self.find_user_tiers()

                self.fov_count = 0
                last_user = min(self.playing_time_list, key=self.playing_time_list.get)
                self.last_seg_idx = max(0, int(np.floor(self.playing_time_list[last_user]/Config.seg_duration)) - Config.server_fov_pre_len)
                
                if Config.show_system:
                    print("Last user playing idx is: ", self.last_seg_idx)
                self.delete_fov_table()
                # Only update per interval for all
                self.update_saliency_map_per_interval()
                ## Modified function here
                # self.update_seg_ave_distribution()
                self.update_seg_ave_distribution_dbscan()

    def get_user_fovs_tmm(self, client_display_idx, req_seg_idx):
        ## Pay attention here, the last dim is expand in the network
        # So generate None*5*5*6 for input (mean/var)
        if Config.fov_debug:
            print("Requested idx: ", req_seg_idx)
            print("head of trace_idx: ", self.last_seg_idx)

        other_mean_dis = np.empty(shape=((0, Config.n_pitch, Config.n_yaw)))
        other_std_dis = np.empty(shape=((0, Config.n_pitch, Config.n_yaw)))

        ava_others = []
        # First of all, check how many other users are available
        for key, value in self.distribution_map.items():
            if value[-1][0] >= req_seg_idx:
                ava_others += [key]

        if not len(ava_others):
            # No others are available
            return other_mean_dis, other_std_dis, 0

        head_of_trace_idx = self.last_seg_idx
        # Get mean and var of other from [display_seg_idx to seg_idx then append zero]
        for i in range(Config.target_user_pred_len):
            curr_seg = client_display_idx + i
            gap = curr_seg - head_of_trace_idx
            if curr_seg > req_seg_idx:
                seg_distrs_mean = np.zeros((1,Config.n_pitch, Config.n_yaw))
                seg_distrs_std = np.zeros((1,Config.n_pitch, Config.n_yaw))
            else:
                seg_distrs = np.empty(shape=((0, Config.n_pitch, Config.n_yaw)))
                for other_id in ava_others:
                    other_distributions = self.distribution_map[other_id]
                    # Use offset to get segment distribution
                    # print(other_distributions[gap][0], curr_seg, gap)
                    # print(len(other_distributions), other_distributions[gap])
                    assert other_distributions[gap][0] == curr_seg
                    # Get this user's dis
                    # print(np.array([np.expand_dims(other_distributions[gap][1], axis=2)]).shape)
                    seg_distrs = np.append(seg_distrs, np.array([other_distributions[gap][1]]), axis=0)
                    # print(seg_distrs.shape)
                # Get all distributions for this segment
                # Get mean and std
                seg_distrs = np.array(seg_distrs)
                seg_distrs_mean = np.mean(seg_distrs, axis=0, keepdims=True)
                assert np.round(np.sum(seg_distrs_mean), 2) == 1.
                seg_distrs_std = np.std(seg_distrs, axis=0, keepdims=True)
            # print(other_mean_dis.shape)
            # print(seg_distrs_mean.shape)
            other_mean_dis = np.append(other_mean_dis, seg_distrs_mean, axis = 0)
            other_std_dis = np.append(other_std_dis, seg_distrs_std, axis= 0 )
        if Config.fov_debug:
            print("other users' distribution mean and std")
            print(other_mean_dis.shape)
            print(other_std_dis.shape)
        # print(other_mean_dis)
        return other_mean_dis, other_std_dis, len(ava_others)

    def get_user_fovs_enhanced(self, display_segs, seg_idx):
        # Get distribution and history fov traces of other users to the target user
        neighbors_trace = []
        distribution_map = []
        nei_target_traces = []
        user_dis_maps = dict()
        if Config.fov_debug:
            print("Requested idx: ", seg_idx)
            print("head of trace_idx: ", self.last_seg_idx)
        head_of_trace_idx = self.last_seg_idx
        for key, value in self.distribution_map.items():
            # For each user
            if value[-1][0] >= seg_idx:
                user_display_trace = []
                user_traces = self.client_fovs[key]
                # assert user_traces[0][0] == head_of_trace_idx
                for display_seg in display_segs:
                    if Config.fov_debug:
                        print("User trace seg idx: ", user_traces[display_seg - head_of_trace_idx][0])
                        print("History fov seg idx : ", display_seg)
                    # assert user_traces[display_seg - head_of_trace_idx][0] == display_seg
                    user_display_trace.append(user_traces[display_seg - head_of_trace_idx])
                neighbors_trace.append([key, user_display_trace])
                # Get distribution map
                # user_distribution = self.distribution_map[key]
                user_distribution = value
                for dis in user_distribution:
                    if dis[0] == seg_idx:
                        # print(dis)
                        user_dis_maps[key] = (dis[0], dis[1], dis[2])
                # Add trace to return in addition to distribution
                nei_target_traces.append([key, user_traces[seg_idx - head_of_trace_idx]])

        # Some addition clustered distributiuon, deprecated
        if seg_idx in self.seg_clustered_distribution:
            ave_seg_dis = self.seg_clustered_distribution[seg_idx]
        else:
            ave_seg_dis = np.zeros((Config.n_pitch, Config.n_yaw))
        return neighbors_trace, user_dis_maps, nei_target_traces, ave_seg_dis


    def get_user_fovs(self, display_segs, seg_idx):
        # Get distribution and history fov traces of other users to the target user
        neighbors_trace = []
        distribution_map = []
        nei_target_traces = []
        user_dis_maps = dict()
        if Config.fov_debug:
            print("Requested idx: ", seg_idx)
            print("head of trace_idx: ", self.last_seg_idx)
        head_of_trace_idx = self.last_seg_idx
        for key, value in self.distribution_map.items():
            # For each user
            if value[-1][0] >= seg_idx:
                user_display_trace = []
                user_traces = self.client_fovs[key]
                # assert user_traces[0][0] == head_of_trace_idx
                for display_seg in display_segs:
                    if Config.fov_debug:
                        print("User trace seg idx: ", user_traces[display_seg - head_of_trace_idx][0])
                        print("History fov seg idx : ", display_seg)
                    # assert user_traces[display_seg - head_of_trace_idx][0] == display_seg
                    user_display_trace.append(user_traces[display_seg - head_of_trace_idx])
                neighbors_trace.append([key, user_display_trace])
                # Get distribution map
                # user_distribution = self.distribution_map[key]
                user_distribution = value
                for dis in user_distribution:
                    if dis[0] == seg_idx:
                        # print(dis)
                        user_dis_maps[key] = (dis[0], dis[1], dis[2])
                # Add trace to return in addition to distribution
                nei_target_traces.append([key, user_traces[seg_idx - head_of_trace_idx]])
        return neighbors_trace, user_dis_maps, nei_target_traces
