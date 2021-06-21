import numpy as np
from config import Config
from utils import *
from scipy.io import loadmat


SHOW_ENTRO = 0
##############################################
##  0,0    0,1    0,2    0,3    0,4    0,5  ##
##  1,0    1,1    1,2    1,3    1,4    1,5  ##
##  2,0    2,1    2,2    2,3    2,4    2,5  ##
##  3,0    3,1    3,2    3,3    3,4    3,5  ##
##  4,0    4,1    4,2    4,3    4,4    4,5  ##
##  5,0    5,1    5,2    5,3    5,4    5,5  ##
##############################################

class User(object):
    def __init__(self, u_id, l_group_idx, init_latency, fov_trace, time_trace, bw_trace, server_seg_idx, v_length, current_time, matrix, tile_map, additional_latency=0, buffer_upper_bound=Config.user_buffer_upper):
        # np.random.seed(u_id)
        self.seg_duration = Config.seg_duration
        self.n_yaw = Config.n_yaw
        self.n_pitch = Config.n_pitch
        self.state = 0              # 0: initial, 1: display 2: freeze
        self.matrix = matrix    #[a,b,weights]
        # self.tile_map = loadmat(Config.tile_map_dir_new)['map']
        # print(self.tile_map[0][0].shape)
        # self.qr_map = loadmat(Config.qr_map_dir)['qq']
        self.u_id = u_id
        self.l_group_idx = l_group_idx
        self.fov_trace_id = fov_trace[0]
        # print('sss', len(fov_trace[1:]))
        self.fov_trace = fov_trace[1:]
        self.bw_trace_id = bw_trace[0]
        self.bw_trace = bw_trace[1:]
        self.v_length = v_length
        self.bw_history = []
        self.bw_his_len = Config.bw_his_len
        self.b_ub = buffer_upper_bound
        # assert server_seg_idx >= init_latency
        self.next_req_seg_idx = server_seg_idx-init_latency + additional_latency    # Modified by enhanced
        self.playing_time = self.next_req_seg_idx*self.seg_duration # + np.random.random()*self.seg_duration
        self.first_fov_to_update_time = 0
        self.buffer_len = 0.0
        self.buffer = []
        self.freezing_tol = Config.freezing_tol
        self.start_up_th = Config.user_start_up_th
        self.time_trace = time_trace[1:]
        ## Enhanced 
        self.time_idx = 1
        self.last_trace_time = self.time_trace[self.time_idx-1] * Config.ms_in_s   # in ms
        #####
        # self.time_idx = np.random.randint(1,len(self.time_trace))
        # self.last_trace_time = self.time_trace[self.time_idx-1] * Config.ms_in_s   # in ms
        self.pre_time_idx = self.time_idx
        self.n_time_loop = -1

        self.downloaded_tiles = []
        self.cross_lists = {}
        self.pre_fetching_time = current_time
        # self.generate_initial_tiles()
        self.request_tile_idx = Config.default_tiles_v1
        self.request_tile_bitrate = Config.default_rates_v1

        self.kf = kalman_filter()

        self.tile_map = tile_map

    def get_buffer_upper_bound(self):
        return self.b_ub

    def generate_initial_tiles(self):
        center = self.fov_trace[self.next_req_seg_idx][1:][int((len(self.fov_trace[self.next_req_seg_idx][1:])-1)/2)][1][:2]
        new_center = [int(cetner[0]/np.pi*180.0+180.0), int(center[1]/np.pi*180.0+90.0)]
        tiles = self.tile_map[new_center[1]][new_center[0]] 
        

    def sim_wait(self, server_encoding_time, wait_time):
        if Config.debug:
            print("###################")
            print("# Sim wait")
            print("###################")
        pre_playing_time = self.playing_time
        return server_encoding_time+wait_time, [[server_encoding_time, server_encoding_time+wait_time, pre_playing_time, pre_playing_time+wait_time, 'p']]

    def wait(self, wait_time): 
        if Config.debug:
            print("###################")
            print("# Waiting")
            print("###################")        
        self.buffer_len -= wait_time
        # assert self.buffer_len > 0.0
        # assert np.round(self.buffer_len) <= Config.user_buffer_upper
        self.playing_time += wait_time
        self.playing_time = np.round(self.playing_time/10)*10
        # print(self.playing_time)
        duration = self.time_trace[self.time_idx] * Config.ms_in_s - self.last_trace_time  # in ms
        if duration > wait_time:
            self.last_trace_time += wait_time
        else:
            temp_wait_time = wait_time
            while duration < temp_wait_time:
                self.last_trace_time = self.time_trace[self.time_idx] * Config.ms_in_s
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0
                    self.n_time_loop += 1
                temp_wait_time -= duration
                duration = self.time_trace[self.time_idx] * Config.ms_in_s - self.last_trace_time
            self.last_trace_time += temp_wait_time
            # assert self.last_trace_time < self.time_trace[self.time_idx] * Config.ms_in_s

    def sim_fetching(self, server_encoding_time):
        ## Record ONLY when state change!!!!!!!!!!!!!!!!
        if Config.debug:
            print("###################")
            print("# Sim seg: ", self.next_req_seg_idx)
            print("###################")  
        tiles_idx = self.request_tile_idx
        tile_rates = self.request_tile_bitrate
        last_event_time = server_encoding_time
        last_ckpt_time = server_encoding_time
        seg_idx = self.next_req_seg_idx
        seg_start_time = seg_idx*self.seg_duration
        event_list = []
        self.rtt = np.random.uniform(Config.rtt_low, Config.rtt_high)          # in ms
        tmp_time_idx = self.time_idx
        tmp_last_trace_time = self.last_trace_time
        tmp_last_playing_time = self.playing_time
        tmp_last_ckpt_playing_time = self.playing_time
        tmp_buffer_len = self.buffer_len
        tmp_freezing_fraction = 0.0
        tmp_state = self.state
        last_n_tile = 0
        # print("Enter sim, state is: ", tmp_state)
        # Get state change
        # Sim rtt 
        if tmp_state == 0 or tmp_state == 2:
            last_ckpt_time += self.rtt
        else:
            if self.rtt >= tmp_buffer_len:
                last_ckpt_time += tmp_buffer_len
                tmp_last_ckpt_playing_time += tmp_buffer_len
                event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                last_event_time = last_ckpt_time
                tmp_last_playing_time = tmp_last_ckpt_playing_time
                tmp_buffer_len = 0.0
                tmp_state = 2
            else:
                last_ckpt_time += self.rtt
                tmp_last_ckpt_playing_time += self.rtt
                tmp_buffer_len -= self.rtt
        duration = self.time_trace[tmp_time_idx] * Config.ms_in_s - tmp_last_trace_time  # in ms
        if duration > self.rtt:
            tmp_last_trace_time += self.rtt
        else:
            temp_rtt = self.rtt
            while duration < temp_rtt:
                tmp_last_trace_time = self.time_trace[tmp_time_idx] * Config.ms_in_s
                tmp_time_idx += 1
                if tmp_time_idx >= len(self.time_trace):
                    tmp_time_idx = 1
                    tmp_last_trace_time = 0.0
                temp_rtt -= duration
                duration = self.time_trace[tmp_time_idx] * Config.ms_in_s - tmp_last_trace_time
            tmp_last_trace_time += temp_rtt
            # assert tmp_last_trace_time < self.time_trace[tmp_time_idx] * Config.ms_in_s
            # Sim rtt done
        # print('Sim after rtt', last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time)
        # Download tiles
        for t_idx in range(len(tiles_idx)):
            if_nece = tiles_idx[t_idx][1]
            if t_idx == len(tiles_idx)-1 or tiles_idx[t_idx+1][1] == 0:
                # Tiles are not necessary, if freeze right now, go to display.
                # Change state
                last_n_tile = 1
            tile_size = Config.tile_ratio*tile_rates[t_idx]*self.seg_duration/Config.ms_in_s         # in mb
            tile_sent = 0.0
            # print('Curr tile is the last n-tile?: ', last_n_tile)
            # print(last_ckpt_time)
            while True:
                throughput = self.bw_trace[tmp_time_idx]                                                # in Mbps
                duration = self.time_trace[tmp_time_idx]*Config.ms_in_s - tmp_last_trace_time           # in ms
                deliverable_size = throughput * duration*Config.packet_payload_portion/Config.ms_in_s             # mbps*s = mb
                # print(tile_size, throughput, duration, deliverable_size)
                if deliverable_size + tile_sent >= tile_size:
                    fraction = (tile_size - tile_sent)/(throughput*Config.packet_payload_portion)*Config.ms_in_s     # in ms
                    # print('fraction is: ', fraction)
                    if tmp_state == 1:
                        # assert tmp_freezing_fraction == 0.0
                        fake_tmp_freezeing_freezing = np.maximum(fraction - tmp_buffer_len, 0.0)       # modified based on playing speed
                        if fake_tmp_freezeing_freezing > 0:
                            # print("There will be freeze!")
                            if if_nece:
                                last_ckpt_time += tmp_buffer_len
                                tmp_last_ckpt_playing_time += tmp_buffer_len
                                tmp_last_trace_time += tmp_buffer_len
                                event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                                last_event_time = last_ckpt_time
                                tmp_last_playing_time = tmp_last_ckpt_playing_time
                                tmp_buffer_len = 0.0
                                if fake_tmp_freezeing_freezing >= self.freezing_tol:
                                    assert 0 == 1
                                    # # Enter freezing from state=1 and then timeout
                                    # last_ckpt_time += self.freezing_tol
                                    # event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                                    # return last_ckpt_time, event_list
                                else:
                                    # Download is finished add freeze and return
                                    last_ckpt_time += fake_tmp_freezeing_freezing
                                    tmp_freezing_fraction += fake_tmp_freezeing_freezing
                                    tmp_last_trace_time += fake_tmp_freezeing_freezing
                                    if last_n_tile:
                                        event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                                        return last_ckpt_time, event_list
                                    tmp_state = 2
                                    break
                            else:
                                assert 0 == 1
                                # # End download when for curr seg
                                # assert np.floor(tmp_last_playing_time) < seg_idx
                                # assert tmp_buffer_len > self.seg_duration
                                # # Terminate u-tiles if seg_idx is same as display
                                # gap_time = seg_start_time - tmp_last_ckpt_playing_time
                                # tmp_last_ckpt_playing_time = seg_start_time
                                # last_ckpt_time += gap_time
                                # tmp_buffer_len -= gap_time
                                # assert np.round(tmp_buffer_len, 3) == self.seg_duration
                                # event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                                # return last_ckpt_time, event_list
                        else:
                            # print("There is no freeze!")
                            # Check if it is n-tile or u-tile
                            if if_nece:
                                last_ckpt_time += fraction
                                tmp_last_ckpt_playing_time += fraction
                                tmp_buffer_len -= fraction
                                tmp_last_trace_time += fraction
                                # assert tmp_buffer_len > 0.0
                                # if last_n_tile:
                                #     event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                                #     tmp_buffer_len += self.seg_duration
                                #     return last_ckpt_time, event_list

                                # No return after last_n_tile
                                if last_n_tile:
                                    tmp_buffer_len += self.seg_duration
                                break
                            else:
                                # assert tmp_last_ckpt_playing_time < seg_start_time
                                if tmp_last_ckpt_playing_time + fraction > seg_start_time:
                                    # Download u-tile for current seg, return
                                    # assert np.floor(tmp_last_ckpt_playing_time/Config.ms_in_s) < seg_idx
                                    # assert tmp_buffer_len > self.seg_duration
                                    gap_time = seg_start_time - tmp_last_ckpt_playing_time
                                    tmp_last_ckpt_playing_time = seg_start_time
                                    last_ckpt_time += gap_time
                                    tmp_buffer_len -= gap_time
                                    # assert np.round(tmp_buffer_len, 3) == self.seg_duration
                                    event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                                    return last_ckpt_time, event_list
                                else:
                                    # Continue download
                                    last_ckpt_time += fraction
                                    tmp_last_ckpt_playing_time += fraction
                                    tmp_buffer_len -= fraction
                                    tmp_last_trace_time += fraction
                                    break

                    elif tmp_state == 2:
                        # assert if_nece
                        if tmp_freezing_fraction + fraction >= self.freezing_tol:
                            # Download n-tiles cause timeout, need to consider
                            gap = self.freezing_tol - tmp_freezing_fraction
                            last_ckpt_time += gap
                            # assert tmp_last_ckpt_playing_time == tmp_last_playing_time
                            # assert np.round(last_ckpt_time - last_event_time, 3) == self.freezing_tol
                            event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                            return last_ckpt_time, event_list
                        else:
                            # Finish download in freeze, if last n-tile, record and return
                            if last_n_tile:
                                last_ckpt_time += fraction
                                tmp_freezing_fraction += fraction
                                event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                                tmp_buffer_len += self.seg_duration
                                # assert np.round(tmp_last_playing_time) == seg_start_time
                                return last_ckpt_time, event_list
                            else:
                                # else keep downloading, it much be n-tile
                                # Continue last event
                                last_ckpt_time += fraction
                                tmp_last_trace_time += fraction
                                tmp_freezing_fraction += fraction
                                # assert tmp_last_ckpt_playing_time == tmp_last_playing_time
                                # assert tmp_buffer_len == 0.0
                                break
                    else:
                        # No timeout
                        # assert if_nece
                        # assert last_event_time == server_encoding_time
                        # assert tmp_last_playing_time == tmp_last_ckpt_playing_time
                        tmp_freezing_fraction += fraction
                        last_ckpt_time += fraction
                        tmp_last_trace_time += fraction
                        if last_n_tile:
                            event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                            return last_ckpt_time, event_list
                        break
                if Config.debug:
                    print("!!!!!!!!!!!!!!!!!!!Change trace time slot!!!!!!!!!!!!!!!!!!!")
                # For current tp record
                # Change tmp_last_trace_time at last
                tile_sent += deliverable_size
                if tmp_state == 1:
                    # assert 0 == 1
                    fake_tmp_freezeing_freezing = np.maximum(duration - tmp_buffer_len, 0.0)       # modified based on playing speed
                    if fake_tmp_freezeing_freezing > 0:
                        if if_nece:
                            last_ckpt_time += tmp_buffer_len
                            tmp_last_ckpt_playing_time += tmp_buffer_len
                            event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                            last_event_time = last_ckpt_time
                            tmp_last_playing_time = tmp_last_ckpt_playing_time
                            tmp_buffer_len = 0.0
                            if fake_tmp_freezeing_freezing >= self.freezing_tol:
                                assert 0 == 1
                                # # Enter freezing from state=1 and then timeout
                                # last_ckpt_time += self.freezing_tol
                                # event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                                # return last_ckpt_time, event_list
                            else:
                                last_ckpt_time += fake_tmp_freezeing_freezing
                                tmp_freezing_fraction += fake_tmp_freezeing_freezing
                                tmp_state = 2
                        else:
                            assert 0 == 1
                            # End download when for curr seg
                            # assert np.floor(tmp_last_playing_time) < seg_idx
                            # assert tmp_buffer_len > self.seg_duration
                            # # Terminate u-tiles if seg_idx is same as display
                            # gap_time = seg_start_time - tmp_last_ckpt_playing_time
                            # tmp_last_ckpt_playing_time = seg_start_time
                            # last_ckpt_time += gap_time
                            # tmp_buffer_len -= gap_time
                            # assert np.round(tmp_buffer_len, 3) == self.seg_duration
                            # event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                            # return last_ckpt_time, event_list
                    else:
                        # Check if it is n-tile or u-tile
                        if if_nece:
                            last_ckpt_time += duration
                            tmp_last_ckpt_playing_time += duration
                            tmp_buffer_len -= duration
                            # assert tmp_buffer_len > 0.0
                            # if last_n_tile and tmp_last_ckpt_playing_time >= (seg_idx-1)*self.seg_duration:
                            #     # if it is last n-tile check buffer, if download curr is same with playing, return
                            #     event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                            #     return last_ckpt_time, event_list
                        else:
                            if tmp_last_ckpt_playing_time + duration > seg_start_time:
                                # Download u-tile for current seg, return
                                # assert np.floor(tmp_last_playing_time/self.seg_duration) < seg_idx
                                # assert tmp_buffer_len > self.seg_duration
                                gap_time = seg_start_time - tmp_last_ckpt_playing_time
                                tmp_last_ckpt_playing_time = seg_start_time
                                last_ckpt_time += gap_time
                                tmp_buffer_len -= gap_time
                                # assert np.round(tmp_buffer_len, 3) == self.seg_duration
                                event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
                                return last_ckpt_time, event_list
                            else:
                                # Continue download
                                last_ckpt_time += duration
                                tmp_last_ckpt_playing_time += duration
                                tmp_buffer_len -= duration
                elif tmp_state == 2:
                    # assert if_nece
                    if tmp_freezing_fraction + duration >= self.freezing_tol:
                        # Download n-tiles cause timeout, need to consider
                        gap = self.freezing_tol - tmp_freezing_fraction
                        last_ckpt_time += gap
                        # assert tmp_last_ckpt_playing_time == tmp_last_playing_time
                        # assert np.round(last_ckpt_time - last_event_time, 3) == self.freezing_tol
                        event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                        return last_ckpt_time, event_list
                    else:
                        last_ckpt_time += duration
                        tmp_freezing_fraction += duration
                        # assert tmp_last_ckpt_playing_time == tmp_last_playing_time
                        # assert tmp_buffer_len == 0.0
                else:
                    # No timeout
                    # assert last_event_time == server_encoding_time
                    # assert tmp_last_playing_time == tmp_last_ckpt_playing_time
                    tmp_freezing_fraction += duration
                    last_ckpt_time += duration
                    # if last_n_tile:
                    #     event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'f'])
                    #     return last_ckpt_time, event_list
                tmp_last_trace_time = self.time_trace[tmp_time_idx]*Config.ms_in_s          # in ms
                tmp_time_idx += 1
                if tmp_time_idx >= len(self.time_trace):
                    tmp_time_idx = 1
                    tmp_last_trace_time = 0.0  # in ms
        event_list.append([last_event_time, last_ckpt_time, tmp_last_playing_time, tmp_last_ckpt_playing_time, 'p'])
        return last_ckpt_time, event_list

    def get_download_tile_info(self):
        return self.request_tile_idx, self.request_tile_bitrate

    def get_download_seg_idx(self):
        return self.next_req_seg_idx

    def fetching(self):
        # This function should be called after tile rate and index are confirmed.
        # If fov information is not enough, operate sim_wait() before this and confirm tile index and rates
        # print("start fetching!")
        if Config.debug:
            print("###################")
            print("# Fetching seg: ", self.next_req_seg_idx)
            print("###################")        
        tiles_idx = self.request_tile_idx
        tile_rates = self.request_tile_bitrate
        # assert len(tiles_idx) == len(tile_rates)
        seg_idx = self.next_req_seg_idx
        seg_start_time = seg_idx*self.seg_duration
        total_freezing_fraction = self.consume_rtt()                           # in ms
        total_downloading_fraction = self.rtt                                  # in ms
        last_n_tile = 0
        curr_seg_tiles_info = [self.rtt]
        # Start download tiles    
        for t_idx in range(len(tiles_idx)):
            if_nece = tiles_idx[t_idx][1]
            if t_idx == len(tiles_idx) -1 or tiles_idx[t_idx+1][1] == 0:
                last_n_tile = 1
            downloading_fraction = 0.0
            freezing_fraction = 0.0
            tile_size = Config.tile_ratio*tile_rates[t_idx]*self.seg_duration/Config.ms_in_s         # in mb
            tile_sent = 0.0
            # print("Tile size is: ", tile_size)
            while True:
                throughput = self.bw_trace[self.time_idx]   # in Mbps
                duration = self.time_trace[self.time_idx]*Config.ms_in_s - self.last_trace_time                     # in ms
                deliverable_size = throughput * duration * Config.packet_payload_portion/Config.ms_in_s             # mbps*s = mb
                # print("Deliverable: ", deliverable_size)
                if deliverable_size + tile_sent >= tile_size:
                    fraction = (tile_size - tile_sent)/(throughput*Config.packet_payload_portion)*Config.ms_in_s     # in ms
                    # print("Could be finished in this time slot.")
                    # print("Fraction is ", fraction)
                    if self.state == 1:
                        # assert freezing_fraction == 0.0
                        temp_freezing = np.maximum(fraction - self.buffer_len, 0.0)       # modified based on playing speed
                        if temp_freezing > self.freezing_tol:
                            assert 0 == 1
                        if not if_nece and self.playing_time + fraction > seg_start_time:
                            # assert self.buffer_len > self.seg_duration
                            gap = seg_start_time - self.playing_time
                            downloading_fraction += gap
                            total_downloading_fraction += gap
                            self.buffer_len -= gap
                            # assert np.round(self.buffer_len, 2) == self.seg_duration
                            self.last_trace_time += gap
                            self.playing_time += gap
                            # assert np.round(self.playing_time, 2) == seg_start_time
                            tile_sent += gap/Config.ms_in_s*throughput*Config.packet_payload_portion
                            curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_sent, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                            self.next_req_seg_idx += 1
                            return curr_seg_tiles_info, total_downloading_fraction
                        if not if_nece:
                            assert temp_freezing == 0.0 
                        downloading_fraction += fraction
                        total_downloading_fraction += fraction
                        freezing_fraction += np.maximum(fraction - self.buffer_len, 0.0)  
                        total_freezing_fraction += np.maximum(fraction - self.buffer_len, 0.0)  
                        self.last_trace_time += fraction
                        self.playing_time += np.minimum(fraction, self.buffer_len)        
                        self.buffer_len = np.maximum(self.buffer_len-fraction, 0.0)                            
                        self.buffer.append([tiles_idx[t_idx], tile_rates[t_idx]])
                        if if_nece and last_n_tile:
                            self.buffer_len += self.seg_duration
                            curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                            if temp_freezing > 0:
                                self.next_req_seg_idx += 1
                                return curr_seg_tiles_info, total_downloading_fraction
                            break
                        curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                        break
                    elif self.state == 2:
                        # assert if_nece
                        # assert self.buffer_len == 0.0
                        if total_freezing_fraction + fraction >= self.freezing_tol:
                            time_out = 1
                            downloading_fraction += self.freezing_tol - total_freezing_fraction
                            total_downloading_fraction += self.freezing_tol - total_freezing_fraction
                            self.last_trace_time += self.freezing_tol - total_freezing_fraction
                            tile_sent += (self.freezing_tol - total_freezing_fraction)/Config.ms_in_s * throughput * Config.packet_payload_portion # in Mb
                            freezing_fraction += self.freezing_tol - total_freezing_fraction
                            total_freezing_fraction = self.freezing_tol
                            self.state = 0
                            # assert tile_sent < tile_size
                            # To be modified
                            curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_sent, downloading_fraction, self.freezing_tol - total_freezing_fraction, self.buffer_len, self.playing_time])
                            return curr_seg_tiles_info, total_downloading_fraction
                        # print(self.buffer_len)
                        # print(self.playing_time)
                        # print(seg_start_time)
                        # assert np.round(self.playing_time) == seg_start_time
                        freezing_fraction += fraction
                        total_freezing_fraction += fraction
                        downloading_fraction += fraction
                        total_downloading_fraction += fraction
                        self.last_trace_time += fraction
                        self.buffer.append([tiles_idx[t_idx], tile_rates[t_idx]])
                        if last_n_tile:
                            self.state = 1
                            self.buffer_len += self.seg_duration
                            curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                            self.next_req_seg_idx += 1
                            return curr_seg_tiles_info, total_downloading_fraction
                            # Whether break current download or continue download all tiles which might not belong to necesary tiles
                        curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                        break
                    else:
                        # assert if_nece
                        # assert self.buffer_len < self.start_up_th
                        downloading_fraction += fraction
                        total_downloading_fraction += fraction
                        freezing_fraction += fraction
                        total_freezing_fraction += fraction
                        self.last_trace_time += fraction
                        self.buffer.append([tiles_idx[t_idx], tile_rates[t_idx]])
                        if last_n_tile:
                            self.buffer_len += self.seg_duration
                            # print(self.buffer_len)
                            curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                            if self.buffer_len >= self.start_up_th:
                                # Because it might happen after one long freezing (not exceed freezing tol)
                                # And resync, enter initial phase
                                buffer_end_time = seg_start_time + self.seg_duration
                                self.playing_time = buffer_end_time - self.buffer_len
                                # print(buffer_end_time, self.buffer)
                                self.state = 1
                            self.next_req_seg_idx += 1
                            return curr_seg_tiles_info, total_downloading_fraction
                        curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_size, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                        break
                if Config.debug:
                    print("!!!!!!!!!!!!!!!!!!!Change trace time slot!!!!!!!!!!!!!!!!!!!")
                # One chunk downloading does not finish
                if self.state == 1:
                    temp_freezing = np.maximum(duration - self.buffer_len, 0.0)       # modified based on playing speed
                    # Freezing time exceeds tolerence
                    if temp_freezing + total_freezing_fraction >= self.freezing_tol:
                        assert 0 == 1
                        # time_out = 1
                        # self.last_trace_time += self.freezing_tol + self.buffer_len
                        # downloading_fraction += self.freezing_tol + self.buffer_len
                        # freezing_fraction = self.freezing_tol - total_freezing_fraction
                        # total_freezing_fraction = self.freezing_tol
                        # self.playing_time += self.buffer_len
                        # tile_sent += (self.freezing_tol + self.buffer_len)/Config.ms_in_s * throughput * PACKET_PAYLOAD_PORTION   # in Mb  
                        # self.buffer_len = 0.0
                        # self.state = 0
                        # # exceed TOL, enter startup, freezing time equals TOL
                        # assert tile_sent < tile_size
                        # curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_sent, downloading_fraction, self.freezing_tol - total_freezing_fraction, self.buffer_len, self.playing_time])
                        # return curr_seg_tiles_info  
                    if not if_nece and self.playing_time + duration > seg_start_time:
                        if Config.debug:
                            print(self.buffer_len)
                        # assert self.buffer_len > self.seg_duration
                        gap = seg_start_time - self.playing_time
                        downloading_fraction += gap
                        total_downloading_fraction += gap
                        self.last_trace_time += gap
                        self.buffer_len -= gap
                        self.playing_time += gap
                        # assert np.round(self.buffer_len,2) == self.seg_duration
                        tile_sent += gap/Config.ms_in_s*throughput*Config.packet_payload_portion
                        curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_sent, downloading_fraction, freezing_fraction, self.buffer_len, self.playing_time])
                        self.next_req_seg_idx += 1
                        return curr_seg_tiles_info, total_downloading_fraction
                    tile_sent += deliverable_size
                    downloading_fraction += duration    # in ms
                    total_downloading_fraction += duration
                    self.playing_time += np.minimum(self.buffer_len, duration)   
                    self.buffer_len = np.maximum(self.buffer_len - duration, 0.0)         
                    # update buffer and state
                    if temp_freezing > 0:
                        # enter freezing
                        # assert if_nece
                        self.state = 2
                        # assert self.buffer_len == 0.0
                        freezing_fraction += temp_freezing
                        total_freezing_fraction += temp_freezing
                # Freezing during trace
                elif self.state == 2:
                    # assert self.buffer_len == 0.0
                    # assert if_nece
                    if duration + total_freezing_fraction >= self.freezing_tol:
                        time_out = 1
                        downloading_fraction += self.freezing_tol - total_freezing_fraction
                        total_downloading_fraction += self.freezing_tol - total_freezing_fraction
                        self.last_trace_time += self.freezing_tol - total_freezing_fraction   # in ms
                        tile_sent += (self.freezing_tol - total_freezing_fraction)/Config.ms_in_s * throughput * Config.packet_payload_portion # in Kbits
                        freezing_fraction += self.freezing_tol - total_freezing_fraction
                        total_freezing_fraction = self.freezing_tol
                        self.state = 0
                        # Download is not finished, chunk_size is not the entire chunk
                        # assert tile_sent < tile_size
                        curr_seg_tiles_info.append([tiles_idx[t_idx], tile_size, tile_sent, downloading_fraction, self.freezing_tol - total_freezing_fraction, self.buffer_len, self.playing_time])
                        return curr_seg_tiles_info, total_downloading_fraction
                    freezing_fraction += duration   # in ms
                    total_freezing_fraction += duration
                    downloading_fraction += duration
                    total_downloading_fraction += duration
                    tile_sent += deliverable_size    # in kbits
                # Startup
                else:
                    assert self.buffer_len < self.start_up_th
                    # if freezing_fraction + duration > self.freezing_tol:
                    #   self.buffer = 0.0
                    #   time_out = 1
                    #   self.last_trace_time += self.freezing_tol - freezing_fraction   # in ms
                    #   downloading_fraction += self.freezing_tol - freezing_fraction
                    #   chunk_sent += (self.freezing_tol - freezing_fraction) * throughput * PACKET_PAYLOAD_PORTION # in Kbits
                    #   freezing_fraction = self.freezing_tol
                    #   # Download is not finished, chunk_size is not the entire chunk
                    # assert chunk_sent < chunk_size
                    #   return chunk_sent, downloading_fraction, freezing_fraction, time_out, start_state
                    tile_sent += deliverable_size
                    downloading_fraction += duration
                    total_downloading_fraction += duration
                    freezing_fraction += duration
                    total_freezing_fraction += duration
                self.last_trace_time = self.time_trace[self.time_idx] * Config.ms_in_s          # in ms
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0  # in ms
                    self.n_time_loop += 1
        self.next_req_seg_idx += 1
        return curr_seg_tiles_info, total_downloading_fraction

    def consume_rtt(self):
        rtt = self.rtt
        freezing_fraction = 0.0
        duration = self.time_trace[self.time_idx] * Config.ms_in_s - self.last_trace_time  # in ms
        if duration > rtt:
            self.last_trace_time += rtt
        else:
            temp_rtt = rtt
            while duration < temp_rtt:
                self.last_trace_time = self.time_trace[self.time_idx] * Config.ms_in_s
                self.time_idx += 1
                if self.time_idx >= len(self.time_trace):
                    self.time_idx = 1
                    self.last_trace_time = 0.0
                temp_rtt -= duration
                duration = self.time_trace[self.time_idx] * Config.ms_in_s - self.last_trace_time
            self.last_trace_time += temp_rtt
            # assert self.last_trace_time < self.time_trace[self.time_idx] * Config.ms_in_s

        if self.state == 1:
            self.playing_time += np.minimum(self.buffer_len, rtt)         # modified based on playing speed, adjusted, * speed
            freezing_fraction += np.maximum(rtt - self.buffer_len, 0.0)   # modified based on playing speed, real time, /speed
            self.buffer_len = np.maximum(0.0, self.buffer_len - rtt)          # modified based on playing speed, adjusted, * speed
            # chech whether enter freezing
            if freezing_fraction > 0.0:
                self.state = 2
        else:
            # If is possible to enter state 2 if player wait and make freeze
            freezing_fraction += rtt
        return freezing_fraction

    def get_id(self):
        return self.u_id

    def get_playing_time(self):
        return self.playing_time

    def get_buffer_length(self):
        return self.buffer_len

    def get_next_req_seg_idx(self):
        return self.next_req_seg_idx

    def predict_bw(self):
        if Config.bw_prediction_method == 0:
            # average
            return np.mean(self.bw_history)

        elif Config.bw_prediction_method == 1:
            # Harmonic mean
            return 0.85*len(self.bw_history)/np.sum([1/bw for bw in self.bw_history])

    def prepare_fov_trace(self):
        curr_playing_seg = int(np.floor(self.playing_time/self.seg_duration))
        curr_seg_trace = self.fov_trace[curr_playing_seg][1:]
        display_segs = [curr_playing_seg]
        if curr_playing_seg >= 1:
            prepared_seg_trace = self.fov_trace[curr_playing_seg-1][1:]
            display_segs.insert(0, curr_playing_seg-1)
        else:
            prepared_seg_trace = []
        prepared_seg_trace.extend(curr_seg_trace)
        # curr_seg_trace = self.fov_trace[curr_playing_seg]
        curr_time = [frame_fov[0] for frame_fov in prepared_seg_trace if frame_fov[0] <= self.playing_time/self.seg_duration]
        curr_seg_yaw = [frame_fov[1][0]/np.pi*180.0+180 for frame_fov in prepared_seg_trace if frame_fov[0] <= self.playing_time/self.seg_duration]
        curr_seg_pitch = [frame_fov[1][1]/np.pi*180.0+90 for frame_fov in prepared_seg_trace if frame_fov[0] <= self.playing_time/self.seg_duration]
        processed_curr_seg_yaw = self.process_degree(curr_seg_yaw)
        return curr_time, processed_curr_seg_yaw, curr_seg_yaw, curr_seg_pitch, display_segs

    def process_degree(self, trace):
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

    def cut_trace(self, trace):
        diff = np.diff(trace)
        if Config.fov_debug:
            print(max(diff))

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

    def adjust_playing_time(self):
        # round to ms level
        self.playing_time = np.round(self.playing_time)
        # print(self.playing_time)

    def tmm_check_target_display(self):
        curr_playing_seg = int(np.floor(self.playing_time/self.seg_duration))
        return curr_playing_seg, self.next_req_seg_idx

    def tmm_get_target_fov_info(self):
        # Get target user previous segment distribution
        curr_playing_seg = int(np.floor(self.playing_time/self.seg_duration))

        target_user_prev_dis = []
        initial_seg = curr_playing_seg - Config.target_his_fov_len
        for i in range(initial_seg, curr_playing_seg):
            if i < 0:
                # less than zero, add zero distribution
                seg_dis = np.zeros((Config.n_pitch, Config.n_yaw))
            else:
                seg_trace = self.fov_trace[i][1:]
                seg_dis = self.get_distribution_from_center(seg_trace)
            target_user_prev_dis += [seg_dis]
        assert len(target_user_prev_dis) == Config.target_his_fov_len
        return np.array(target_user_prev_dis)

    def predict_fov(self):
        ###!!! important processed_yaw_trace is onlye used in individual prediction kf or truncated linear, do NOT return
        time_trace, processed_yaw_trace, yaw_trace, pitch_trace, display_segs = self.prepare_fov_trace()
        # yaw_cut = self.cut_trace(yaw_trace)
        # pitch_cut = self.cut_trace(pitch_trace)
        # The gap between current display time and middle frame of download seg
        prediction_gap = self.next_req_seg_idx * self.seg_duration 
        assert prediction_gap >= self.playing_time
        # fov_gap_within_seg = self.seg_duration/Config.num_fov_per_seg
        gaps = []
        interval = 1/(Config.num_interval+1)
        for i in range(Config.num_interval):
            gaps += [(prediction_gap+(i+1)*interval*self.seg_duration)/Config.ms_in_s]
        # print(self.next_req_seg_idx, gaps)

        #     gap1 = (prediction_gap+0.25*self.seg_duration)/Config.ms_in_s
        # gap2 = (prediction_gap+0.5*self.seg_duration)/Config.ms_in_s
        # gap3 = (prediction_gap+0.75*self.seg_duration)/Config.ms_in_s

        # Test kf
        self.kf.set_traces(time_trace, processed_yaw_trace, pitch_trace)
        self.kf.init_kf()
        centers = []
        if Config.kf_predict:
            predicted_Xs = self.kf.kf_run(gaps[len(gaps)//2], True)
            predicted_center = (predicted_Xs[0][0]%360.0, predicted_Xs[1][0]%180.0)
        else:
            # Only use kf to get rid of noise
            modified_Xs = self.kf.kf_run(gaps[len(gaps)//2])

            # print(time_trace, modified_Xs)
            # input()
            # print(modified_Xs)
            if len(modified_Xs) > 10:
                for i in range(Config.num_interval):
                    centers += [truncated_linear(gaps[i], time_trace, modified_Xs)]
            else:
                centers = [(yaw_trace[-1], pitch_trace[-1]) for _ in range(Config.num_interval)]
                # predicted_center1 = (yaw_trace[-1], pitch_trace[-1])
                # predicted_center2 = (yaw_trace[-1], pitch_trace[-1])
                # predicted_center3 = (yaw_trace[-1], pitch_trace[-1])

        return [time_trace, yaw_trace, pitch_trace], self.next_req_seg_idx, display_segs, centers, (prediction_gap - self.playing_time)/self.seg_duration + 0.5

    def check_ending(self):
        if self.next_req_seg_idx > self.v_length:
            return True
        return False

    def udpate_prefetching_time(self, current_time):
        self.pre_fetching_time = current_time

    def get_prefetching_time(self):
        return self.pre_fetching_time

    def get_sim_bw_trace(self):
        if not os.path.isdir(Config.info_data_path + 'bw/'):
            os.makedirs(Config.info_data_path + 'bw/')
        log_path = Config.info_data_path + 'bw/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        simed_bw_trace = []
        simed_time_trace = []
        loop_time = self.time_trace[-1]
        # Get first interate
        simed_bw_trace.extend(self.bw_trace[self.pre_time_idx:])
        simed_time_trace.extend(self.time_trace[self.pre_time_idx:])
        while self.n_time_loop > 0:
            simed_bw_trace.extend(self.bw_trace)
            simed_time_trace.extend([time+loop_time for time in self.time_trace])
            self.n_time_loop -= 1
            loop_time += self.time_trace[-1]
        simed_bw_trace.extend(self.bw_trace[:self.time_idx])
        simed_time_trace.extend([loop_time+time for time in self.time_trace[:self.time_idx]])
        assert len(simed_time_trace) == len(simed_bw_trace)
        log_write = open(log_path, 'w')
        for i in range(len(simed_time_trace)):
            log_write.write( str(simed_time_trace[i]) + ' ' + str(simed_bw_trace[i]))
            log_write.write('\n')
        return simed_time_trace, simed_bw_trace

    def update_bw_his(self, tile_info):
        if len(tile_info) > 0:
            total_tile_size = np.sum([tile[2] for tile in tile_info[1:]])
            total_time = np.sum([tile[3] for tile in tile_info[1:]])
            if total_time == 0:
                print(tile_info)
                input()
            ave_bw = total_tile_size/total_time*Config.ms_in_s
            
            # assert len(self.bw_history) <= self.bw_his_len
            if len(self.bw_history) == self.bw_his_len:
                del self.bw_history[0]
            self.bw_history.append(ave_bw)
            if Config.bw_debug:
                print("Ave bandwidth: ", ave_bw, " Mbps")
                print("Bandwidth history: ", self.bw_history)
        else:
            if Config.bw_debug:
                print("Download failed, tile info empty!!", tile_info)

    def filter_fov_info(self, seg_idx):
        # Find specific fov points (frames) of one video segment to upload
        # fov_gap = int(len(self.fov_trace[seg_idx][1:])/(Config.num_fov_per_seg))
        # return [self.fov_trace[seg_idx][1:][int((fov_idx+0.5)*fov_gap)][1][:2] for fov_idx in range(Config.num_fov_per_seg)]

        # Return all info to server
        # print(seg_idx)
        # print(len(self.fov_trace))
        return [fov_info for fov_info in self.fov_trace[seg_idx][1:]]        

    def generate_fov_info(self):
        fov_info = []
        # if Config.debug:
        #     print(self.fov_trace[self.first_fov_to_update_time][0])
        #     print(np.floor(self.first_fov_to_update_time))
        # assert self.fov_trace[self.first_fov_to_update_time][0] == np.floor(self.first_fov_to_update_time)
        for i in range(self.first_fov_to_update_time, int(np.floor(self.playing_time/Config.ms_in_s))):
            # Process 
            fov_info.append([i, self.filter_fov_info(i)])
        self.first_fov_to_update_time = int(np.floor(self.playing_time/Config.ms_in_s))
        return fov_info

    def choose_tiles(self, predicted_seg_idx, predicted_bw, prepared_info, predicted_center, neighbor_fovs, distribution, gap_in_s, nei_target_traces, seg_ave_from_server):
        ff_distribution = None
        p_type = None
        if Config.coordinate_fov_prediction and len(neighbor_fovs) > 0:
            # Using neighbors info
            if Config.choose_tile_debug:
                print("Predicted seg idx: ", predicted_seg_idx)
                print("Predicted bw: ", predicted_bw)
                print("Prepared time: ", prepared_info[0])
                print("Prepared yaw: ", prepared_info[1])
                print("Prepared pitch: ", prepared_info[2])
                print("Predicted cetner: ", predicted_center)
                print("Current display time: ", self.playing_time)
                print("Neighbors' fov: ", neighbor_fovs)
                print("Neighbors' distribution: ", distribution)
            # First of all, find neighbors
            # combined_trace is still (yaw, pitch) order
            combined_trace = [(prepared_info[1][i]/180.0*np.pi, prepared_info[2][i]/180.0*np.pi) for i in range(len(prepared_info[0]))]
            # Cut neighbors' fov
            similarity = []
            # Swith between new versions
            # Not using average, based on distance
            if Config.new_prediction_version == 0 or Config.new_prediction_version == 2 or Config.new_prediction_version == 10:
                for user in neighbor_fovs:
                    # Calculate simularity using fov trace
                    # V0(mmsys) and v2(time)
                    # This is geting history trace to calculate distance
                    user_fov_trace = [(frame_info[1][0]+np.pi, frame_info[1][1]+0.5*np.pi) for seg_info in user[1] for frame_info in seg_info[1] if frame_info[0] <= self.playing_time/Config.ms_in_s]
                    user_time_trace = [frame_info[0] for seg_info in user[1] for frame_info in seg_info[1] if frame_info[0] <= self.playing_time/Config.ms_in_s]
                    if Config.choose_tile_debug:
                        print("Combined trace: ", combined_trace)
                        print("Other users trace", user_fov_trace)
                        print("Target user time: ", prepared_info[0])
                        print("Other user time: ", user_time_trace)
                    # Cut into same length
                    # assert np.abs(len(combined_trace) - len(user_fov_trace)) < 10
                    while len(combined_trace) > len(user_fov_trace):
                        del combined_trace[0]
                    while len(combined_trace) < len(user_fov_trace):
                        del user_fov_trace[0]
                    if len(combined_trace) > Config.distance_tth:
                        user_sim, user_path = calculate_curve_distance(combined_trace[-Config.distance_tth:], user_fov_trace[-Config.distance_tth:])
                    else:
                        user_sim, user_path = calculate_curve_distance(combined_trace, user_fov_trace)
                    if Config.choose_tile_debug:
                        print("Distance (normalized) between user curve is: ", user_sim/len(combined_trace))
                    similarity.append([user[0], user_sim])
                if Config.neighbors_show:
                    print(user_sim)
                    if user_sim > 0 and user_sim < 0.5:
                        plt.scatter([p[0] for p in user_fov_trace], [p[1] for p in user_fov_trace], c='r')
                        plt.scatter([p[0] for p in combined_trace], [p[1] for p in combined_trace], c='b')
                        plt.show()
                        input()
            else:
                if Config.new_prediction_version == 1 or Config.new_prediction_version == 3 :
                    # Don't calculate on client, weight is calculated at server
                    # The weight is calculated on server, stored in nei_distribution[u_id][2]
                    # Copy the weight
                    for key, value in distribution.items():
                        # print(key)
                        # print(value)
                        similarity.append([key, value[2]])
                # Fileter neighbors and then average
                # Process beighbor_distribution based on nei_target_trace
                # To get fov for current seg (the predicted seg)
                # if len(nei_target_traces) >= Config.weight_tth:
                #     yaw_as = []
                #     pitch_as = []
                #     for user_fov_id in range(len(nei_target_traces)):
                #         user = nei_target_traces[user_fov_id]
                #         user_key = user[0]
                #         user_fov_trace = [(frame_info[1][0]+np.pi, frame_info[1][1]+0.5*np.pi) for frame_info in user[1]]
                #         user_time_trace = [frame_info[0] for frame_info in user[1]]
                #         diff = np.diff(user_fov_trace)
                #         yaw_diff_diff = np.mean(np.diff(diff[0]))
                #         pitch_diff_diff = np.mean(np.diff(diff[1]))  # accelerate
                #         yaw_ass.append([user_key, yaw_diff_diff])
                #         pitch_as.append([user_key, pitch_diff_diff])
                #     yaw_a_mean = np.mean([x[1] for x in yaw_as])
                #     yaw_a_std = np.std([x[1] for x in yaw_as])
                #     pitch_as_mean = np.mean([x[1] for x in pitch_as])
                #     pitch_as_std = np.std([x[1] for x in pitch_as])
                # else:
                #     for user in neighbor_fovs:
                #         similarity.append([user[0], user_sim])


            # Find users with short distance
            if Config.show_system:
                print("Target user is %i, and there are %i neighbors." % (self.u_id, len(similarity)))
            if Config.new_prediction_version <= 3 or Config.new_prediction_version == 10:
                alpha, betas, real_betas = self.filter_neighbors(similarity)
                if alpha >= 0.9:
                    # No neighbor
                    p_type = 3
                    ff_distribution = self.find_tiles_indep(predicted_bw, predicted_center)
                else:
                    # Has neighbors
                    p_type = 4
                    # print(alpha, betas, gap_in_s)
                    ff_distribution =  self.find_tiles_jointly(predicted_bw, predicted_center, alpha, betas, distribution, gap_in_s)
        
            else:
                # Ave is generated from server, directly use it

                ff_distribution = self.find_tile_enhanced(predicted_bw, predicted_center, seg_ave_from_server, gap_in_s)
        else:
            # Directely use predicted fov center
            p_type = 3
            ff_distribution = self.find_tiles_indep(predicted_bw, predicted_center)
            real_betas = []
        
        self.record_pred_distribution(ff_distribution, gap_in_s, p_type)
        return

    def record_pred_distribution(self, ff_distribution, gap_in_s, ptype):
        self.cross_lists[self.next_req_seg_idx] = [ff_distribution, gap_in_s, ptype]

    def filter_neighbors(self, similarity):
        # Find neighbors, and return the users will be used for final decision (each individual weight) and outer weight
        similarity.sort(key=lambda x:x[1])
        if Config.show_system:
            print("Similarity between neighbors: ", similarity)
        # closest_neighbors = [nb for nb in similarity if nb[1] <= Config.neighbor_dis_tth]
        # if len(closest_neighbors) > Config.neighbors_upperbound:
        #     closest_neighbors = closest_neighbors[:Config.neighbors_upperbound]
        total_beta = np.sum([get_weight(nb[1]) for nb in similarity])
        betas = [(nb[0], get_weight(nb[1])/total_beta) for nb in similarity]
        real_betas = [(nb[0], get_weight(nb[1])) for nb in similarity]
        # alpha = 1/(1+len(closest_neighbors))
        alpha = 1/(1+total_beta)

        return alpha, betas, real_betas

    def find_tile_enhanced(self, predicted_bw, target_center, seg_ave_from_server, gap_in_s):
        # target_tile_distribution = np.array(self.tile_map[int(target_center[1])][int(target_center[0])])
        [c1, c2, c3] = target_center
        tile_distribution1 = self.tile_map[int(c1[1])][int(c1[0])]
        tile_distribution2 = self.tile_map[int(c2[1])][int(c2[0])]
        tile_distribution3 = self.tile_map[int(c3[1])][int(c3[0])]
        target_tile_distribution = np.multiply(tile_distribution1, 1/3) + np.multiply(tile_distribution2, 1/3) + np.multiply(tile_distribution3, 1/3)
        # target_tile_distribution = tile_distribution2

        target_tile_distribution /= np.sum(target_tile_distribution)
        if np.sum(seg_ave_from_server[1]) == 0:
            return target_tile_distribution
        if Config.new_prediction_version == 5:
            time_w = get_time_weight(gap_in_s)
        else:
            time_w = 1.
        final_tile_distribution = np.multiply(target_tile_distribution, time_w/(seg_ave_from_server[0]+time_w)) + np.multiply(seg_ave_from_server[1], seg_ave_from_server[0]/(seg_ave_from_server[0]+time_w))
        self.rate_allocation(final_tile_distribution, predicted_bw)
        return final_tile_distribution

    def find_tiles_jointly(self, predicted_bw, target_center, alpha, betas, distribution, gap_in_s):
        if Config.new_prediction_version == 2 or Config.new_prediction_version == 3:
            # Use time weight
            # v1: 
            assert gap_in_s >= 0
            time_w = get_time_weight(gap_in_s)
            alpha *= time_w
        alpha = max(0.8, alpha)
            # print(time_w, alpha)
        # target_tile_distribution = np.array(self.tile_map[int(target_center[1])][int(target_center[0])])
        tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
        for c in target_center:
            tile_distribution += self.tile_map[int(c[1])][int(c[0])]/np.sum(self.tile_map[int(c[1])][int(c[0])])
        
        # target_tile_distribution = np.multiply(tile_distribution, 1/len(target_center))
        target_tile_distribution = tile_distribution
        target_tile_distribution /= np.sum(target_tile_distribution)
        # assert len(betas) > 0
        others_tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
        for nb in betas:
            others_tile_distribution += np.multiply(np.array(distribution[nb[0]][1]), nb[1])
        others_tile_distribution /= np.sum(others_tile_distribution)
        final_tile_distribution = np.multiply(target_tile_distribution, alpha) + np.multiply(others_tile_distribution, 1-alpha)
        # print(target_tile_distribution, 'k', others_tile_distribution)
        assert np.round(np.sum(final_tile_distribution),1) == 1
        print(alpha)
        # print(alpha)
        # print(betas)
        # Then choose rate based on the tile distribution
        # To do
        self.rate_allocation(final_tile_distribution, predicted_bw)
        return final_tile_distribution

    def rate_allocation(self, final_tile_distribution, predicted_bw):
        # Assign idx and rate to self.request_tile_idx/self.request_tile_bitrate
        seg_tiles = []
        tmp_rates = []
        if Config.joint_rate_allocation_method == 0:
            for pitch_idx in range(len(final_tile_distribution)):
                for yaw_idx in range(len(final_tile_distribution[pitch_idx])):
                    ratio_bw = final_tile_distribution[pitch_idx][yaw_idx]*predicted_bw/Config.tile_ratio
                    if ratio_bw < 0.25*Config.bitrate[0]:
                        # No rate for this tile
                        continue
                    r_idx = len(Config.bitrate) - 1
                    while r_idx >= 0:
                        if Config.bitrate[r_idx] <= ratio_bw:
                            break
                        else:
                            r_idx -= 1
                    r_idx = max(0, r_idx)
                    seg_tiles.append([(pitch_idx, yaw_idx), 1])
                    tmp_rates.append(Config.bitrate[r_idx])
        elif Config.joint_rate_allocation_method == 1:
            # Maximize sum of qualities of all tiles
            # print(self.matrix[1].shape, self.matrix[2].shape, final_tile_distribution.shape)

            # # Budget
            # min_val = float('inf')
            # for y in final_tile_distribution:
            #     for x in y:
            #         if x > 0 and x < min_val:
            #             min_val = x
            # budget = 10
            # for pitch_idx in range(len(final_tile_distribution)):
            #     if budget <= 0:
            #         break
            #     for yaw_idx in range(len(final_tile_distribution[pitch_idx])):
            #         if final_tile_distribution[pitch_idx][yaw_idx] == 0:
            #             r = np.random.random()
            #             if r <= 0.75:
            #                 if budget > 0:
            #                     final_tile_distribution[pitch_idx][yaw_idx] = min_val
            #                     budget -= 1
            #                 else:
            #                     break
            #             else:
            #                 continue


            confused_mat = np.multiply(np.multiply(final_tile_distribution, self.matrix[1]), self.matrix[2])  #16*32 matrix
            # print(confused_mat)
            total_w = np.sum(confused_mat)

            for pitch_idx in range(len(confused_mat)):
                for yaw_idx in range(len(confused_mat[pitch_idx])):
                    ratio_bw = confused_mat[pitch_idx][yaw_idx]*predicted_bw/Config.tile_ratio/total_w
                    if ratio_bw < 0.05*Config.bitrate[0]:
                        # No rate for this tile
                        continue
                    # if ratio_bw == 0:
                    #     continue
                    # close_rate = [np.abs(ratio_bw-br) for br in Config.bitrate]
                    # r_idx = close_rate.index(min(close_rate))
                    # seg_tiles.append([(pitch_idx, yaw_idx), 1])
                    # tmp_rates.append(Config.bitrate[r_idx])
                    r_idx = len(Config.bitrate) - 1
                    while r_idx >= 0:
                        if Config.bitrate[r_idx] <= ratio_bw:
                            break
                        else:
                            r_idx -= 1
                    r_idx = max(0, r_idx)
                    seg_tiles.append([(pitch_idx, yaw_idx), 1])
                    tmp_rates.append(Config.bitrate[r_idx])

        self.request_tile_idx = seg_tiles
        self.request_tile_bitrate = tmp_rates
        # print(self.request_tile_idx)
        # print(self.request_tile_bitrate)

    # def rate_allocation_log(self, final_tile_distribution, predicted_bw):
    #     ## We have a+blog(r) for each tile and also the weights of each tile
    #     # Assign idx and rate to self.request_tile_idx/self.request_tile_bitrate
    #     seg_tiles = []
    #     tmp_rates = []
    #     if Config.joint_rate_allocation_method == 0:
    #         for pitch_idx in range(len(final_tile_distribution)):
    #             for yaw_idx in range(len(final_tile_distribution[pitch_idx])):
    #                 ratio_bw = final_tile_distribution[pitch_idx][yaw_idx]*predicted_bw/Config.tile_ratio
    #                 if ratio_bw < 0.5*Config.bitrate[0]:
    #                     # No rate for this tile
    #                     continue
    #                 close_rate = [np.abs(ratio_bw-br) for br in Config.bitrate]
    #                 r_idx = close_rate.index(min(close_rate))
    #                 seg_tiles.append([(pitch_idx, yaw_idx), 1])
    #                 tmp_rates.append(Config.bitrate[r_idx])
    #     self.request_tile_idx = seg_tiles
    #     self.request_tile_bitrate = tmp_rates

    def find_tiles_indep(self, predicted_bw, center):
        if Config.show_system:
            print("Indep predict bw is: ", predicted_bw)
            print("Indep center is: ", center)
        effective_tile = []
        effective_tile_rate = []
        ff_dis = None
        if Config.indep_rate_allocation_method == 0:
            # Find all tiles and equally assign rate
            tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
            for c in center:
                tile_distribution += self.tile_map[int(c[1])][int(c[0])]/np.sum(self.tile_map[int(c[1])][int(c[0])])
            
            tile_distribution = np.multiply(tile_distribution, 1/len(center))
            # tile_distribution = tile_distribution2

            ff_dis = tile_distribution / np.sum(tile_distribution)
            effective_tile = [[(pitch_idx, yaw_idx), 1] for pitch_idx in range(len(tile_distribution)) for yaw_idx in range(len(tile_distribution[pitch_idx])) if np.round(tile_distribution[pitch_idx][yaw_idx], 2) > 0]
            effective_tile.sort(key=lambda x:x[0])
            tile_rate = None
            for i in reversed(range(len(Config.bitrate))):
                if Config.bitrate[i]*len(effective_tile)*Config.tile_ratio <= predicted_bw:
                    tile_rate = Config.bitrate[i]
                    break
            if not tile_rate: tile_rate = Config.bitrate[0]
            effective_tile_rate = [tile_rate for i in range(len(effective_tile))]

        elif Config.indep_rate_allocation_method == 1:
            tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
            for c in center:
                tile_distribution += self.tile_map[int(c[1])][int(c[0])]/np.sum(self.tile_map[int(c[1])][int(c[0])])
            
            tile_distribution = np.multiply(tile_distribution, 1/len(center))

            # tile_distribution = self.tile_map[int(center[1])][int(center[0])]
            tile_distribution /= np.sum(tile_distribution)
            ff_dis = tile_distribution
            e_tile_dis = [[(pitch_idx, yaw_idx), tile_distribution[pitch_idx][yaw_idx]] for pitch_idx in range(len(tile_distribution)) for yaw_idx in range(len(tile_distribution[pitch_idx])) if np.round(tile_distribution[pitch_idx][yaw_idx], 2) > 0]
            for e_tile in e_tile_dis:
                ratio_bw = e_tile[1]*predicted_bw/Config.tile_ratio
                if ratio_bw < 0.5*Config.bitrate[0]:
                    # No rate for this tile
                    continue
                close_rate = [np.abs(ratio_bw-br) for br in Config.bitrate]
                r_idx = close_rate.index(min(close_rate))
                effective_tile.append([e_tile[0], 1])
                effective_tile_rate.append(Config.bitrate[r_idx])

        # FOr TMM
        elif Config.indep_rate_allocation_method == 2:
            # tile_distribution = self.tile_map[int(center[1])][int(center[0])]
            tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
            for c in center:
                tile_distribution += self.tile_map[int(c[1])][int(c[0])]/np.sum(self.tile_map[int(c[1])][int(c[0])])

            # min_val = float('inf')
            # for y in tile_distribution:
            #     for x in y:
            #         if x > 0 and x < min_val:
            #             min_val = x
            # budget = 50
            # for pitch_idx in range(len(tile_distribution)):
            #     if budget < 0:
            #         break
            #     for yaw_idx in range(len(tile_distribution[pitch_idx])):
            #         if tile_distribution[pitch_idx][yaw_idx] == 0:
            #             r = np.random.random()
            #             if r <= 0.75:
            #                 if budget > 0:
            #                     tile_distribution[pitch_idx][yaw_idx] = min_val
            #                     budget -= 1
            #                 else:
            #                     break
            #             else:
            #                 continue

            # tile_distribution = np.multiply(tile_distribution, 1/len(center))
            tile_distribution /= np.sum(tile_distribution)
            ff_dis = tile_distribution
            confused_mat = np.multiply(np.multiply(tile_distribution, self.matrix[1]), self.matrix[2])  #16*32 matrix
            total_w = np.sum(confused_mat)
            for pitch_idx in range(len(confused_mat)):
                for yaw_idx in range(len(confused_mat[pitch_idx])):
                    ratio_bw = confused_mat[pitch_idx][yaw_idx]*predicted_bw/Config.tile_ratio/total_w
                    if ratio_bw < 0.25*Config.bitrate[0]:
                        continue
                    
                    # close_rate = [np.abs(ratio_bw-br) for br in Config.bitrate]
                    # r_idx = close_rate.index(min(close_rate))
                    # effective_tile.append([(pitch_idx, yaw_idx), 1])
                    # effective_tile_rate.append(Config.bitrate[r_idx])

                    r_idx = len(Config.bitrate) - 1
                    while r_idx >= 0:
                        if Config.bitrate[r_idx] <= ratio_bw:
                            break
                        else:
                            r_idx -= 1
                    r_idx = max(0, r_idx)
                    effective_tile.append([(pitch_idx, yaw_idx), 1])
                    effective_tile_rate.append(Config.bitrate[r_idx])

            ## Debug
            # size = 0 
            # for i in range(len(effective_tile)):
            #     r =  effective_tile_rate[i]
            #     size += r*Config.tile_ratio
            # print(predicted_bw, size, 'Compare size and bw')

        self.request_tile_idx = effective_tile
        self.request_tile_bitrate = effective_tile_rate
        # print(self.request_tile_idx)
        # print(self.request_tile_bitrate)
        return ff_dis

    def record_downloaded_tiles(self, seg_info):
        self.downloaded_tiles.append(seg_info)

    def save_info(self):
        log_path = Config.info_data_path + 'user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        np.savetxt(log_path, self.downloaded_tiles, fmt='%s')
        return        

    def save_cross_entropy(self, entropy):
        if not os.path.isdir(Config.info_data_path + 'entropy/'):
            os.makedirs(Config.info_data_path + 'entropy/')
        log_path = Config.info_data_path + 'entropy/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        entropy_write = open(log_path, 'w')
        for i in range(len(entropy)):
            entropy_write.write( str(entropy[i][0]) + ' ' + str(entropy[i][1]) + ' ' + str(entropy[i][2]) + ' ' + str(entropy[i][3]) + ' ' + str(entropy[i][4]))
            entropy_write.write('\n')

    def save_fov(self):
        if not os.path.isdir(Config.info_data_path + 'fov/'):
            os.makedirs(Config.info_data_path + 'fov/')
        log_path = Config.info_data_path + 'fov/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        fov_write = open(log_path, 'w')
        for i in range(len(self.fov_trace)):
            frames = self.fov_trace[i][1:]
            id1, id2 = len(frames)//3, len(frames)//3*2
            t1, (n1, n2, n3) = frames[id1]
            tt1, (nn1, nn2, nn3) = frames[id2]
            fov_write.write(str(t1) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3))
            fov_write.write('\t')
            fov_write.write(str(t1) + ' ' + str(n1) + ' ' + str(n2) + ' ' + str(n3))
            fov_write.write('\n')

    def save_rate(self, rates):
        if not os.path.isdir(Config.info_data_path + 'rate/'):
            os.makedirs(Config.info_data_path + 'rate/')
        log_path = Config.info_data_path + 'rate/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        rate_write = open(log_path, 'w')
        for i in range(len(rates)):
            rate_write.write( str(rates[i][0]) + ' ' + str(rates[i][1]))
            rate_write.write('\n')

    def save_wspsnr(self, wspsnr):
        if not os.path.isdir(Config.info_data_path + 'wspsnr/'):
            os.makedirs(Config.info_data_path + 'wspsnr/')
        log_path = Config.info_data_path + 'wspsnr/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        rate_write = open(log_path, 'w')
        for i in range(len(wspsnr)):
            rate_write.write( str(wspsnr[i][0]) + ' ' + str(wspsnr[i][1]))
            rate_write.write('\n')

    def save_ratios(self, ratios):
        if not os.path.isdir(Config.info_data_path + 'ratios/'):
            os.makedirs(Config.info_data_path + 'ratios/')
        log_path = Config.info_data_path + 'ratios/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        rate_write = open(log_path, 'w')
        for i in range(len(ratios)):
            rate_write.write( str(ratios[i][0]) + ' ' + str(ratios[i][1])  + ' ' + str(ratios[i][2]) + ' ' + str(ratios[i][3]))
            rate_write.write('\n')

    def save_nss(self, nss):
        if not os.path.isdir(Config.info_data_path + 'nss/'):
            os.makedirs(Config.info_data_path + 'nss/')
        log_path = Config.info_data_path + 'nss/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        rate_write = open(log_path, 'w')
        for i in range(len(nss)):
            rate_write.write( str(nss[i][0]) + ' ' + str(nss[i][1]))
            rate_write.write('\n')

    def save_freeze(self, freeze):
        if not os.path.isdir(Config.info_data_path + 'freeze/'):
            os.makedirs(Config.info_data_path + 'freeze/')
        log_path = Config.info_data_path + 'freeze/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        freeze_write = open(log_path, 'w')
        for i in range(len(freeze)):
            freeze_write.write( str(freeze[i][0]) + ' ' + str(freeze[i][1]))
            freeze_write.write('\n')

    def save_buffer_len(self, buffer_len):
        if not os.path.isdir(Config.info_data_path + 'buffer/'):
            os.makedirs(Config.info_data_path + 'buffer/')
        log_path = Config.info_data_path + 'buffer/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        buffer_write = open(log_path, 'w')
        for i in range(len(buffer_len)):
            buffer_write.write( str(buffer_len[i][0]) + ' ' + str(buffer_len[i][1]))
            buffer_write.write('\n')

    def save_download_size(self, download_size):
        if not os.path.isdir(Config.info_data_path + 'size/'):
            os.makedirs(Config.info_data_path + 'size/')
        log_path = Config.info_data_path + 'size/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
        size_write = open(log_path, 'w')
        for i in range(len(download_size)):
            size_write.write( str(download_size[i][0]) + ' ' + str(download_size[i][1]))
            size_write.write('\n')

    def evaluate_performance(self):
        # Info is stored in self.downloaded_tiles
        # Save info first of all
        # self.save_info()
        seg_idx = None
        last_timeout_seg = None
        video_rates = []
        video_freeze = []
        cross_entropys = []
        download_sizes = []
        viewed_wspsnr = []
        viewed_ratios = []

        video_nss = []
        overlap_ratio = []

        buffer_initial_time = 0.0
        buffer_record = [(buffer_initial_time, 0.0)]
        print('len of cross list: ', len(self.cross_lists))
        # print(self.downloaded_tiles)
        for seg_info in self.downloaded_tiles:
            seg_idx = seg_info[0]
            download_info = seg_info[1]
            rates = seg_info[2]
            download_time = download_info[0] + np.sum([tile[3] for tile in download_info[1:]])
            download_size = np.sum([tile[2] for tile in download_info[1:]])
            buffer_initial_time += download_time
            current_buffer_len = download_info[-1][-2]
            buffer_record.append((buffer_initial_time, current_buffer_len))
            freezing_time = np.round(np.sum([tile[4] for tile in download_info[1:]])/Config.ms_in_s, 2)
            if freezing_time == self.freezing_tol/Config.ms_in_s:
                last_timeout_seg = seg_idx
                continue
            download_sizes.append((seg_idx, download_size))
            seg_tiles = {}
            for i in range(1, len(download_info)):
                # Translate [(pitch, yaw), 1] to float pitch.yaw
                # assert download_info[i][0][0][0]+download_info[i][0][0][1]*0.1 not in seg_tiles
                seg_tiles[(download_info[i][0][0][0], download_info[i][0][0][1])] = rates[i-1]
            # print(len(download_info), ' tiles are downloaded for seg ', seg_idx)
            # print(seg_tiles)
            frame_info = self.fov_trace[seg_idx]
            # assert frame_info[0] == seg_idx
            n_frames = len(frame_info)-1
            # print('number of frames, ', n_frames)
            seg_rates = 0.0
            frame_weight = 1.0/n_frames 

            seg_nss = 0.0   
            scanpath = np.zeros((Config.n_pitch, Config.n_yaw))
            total_sp = 0
            ### FOR TMM
            seg_quality = 0
            ground_tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
            counter = 0

            # To get nss get a normalized distribution to zero mean and unit derivation

            # for ratios
            # seg_ratios = []
            for f_idx in range(1, len(frame_info)):
                # For TMM to calculate frame psnr
                frame_tile_count = 0
                frame_quality = 0
                frame_view_weight = 0
                gt_yaw, gt_pitch = int((frame_info[f_idx][1][0]/np.pi*180.0+180.0)%360), int(frame_info[f_idx][1][1]/np.pi*180.0+90.0)
                row_idx, col_idx = int(gt_pitch/180*Config.n_pitch), int(gt_yaw/360*Config.n_yaw)
                scanpath[row_idx][col_idx] += 1
                total_sp += 1
                frame_tiles = self.tile_map[gt_pitch][gt_yaw]

                # calculate ratios
                # if seg_idx in self.cross_lists:
                #     frame_ratio = np.sum(self.cross_lists[seg_idx][:]*frame_tiles)
                # else:
                #     frame_ratio = 0
                # seg_ratios += [frame_ratio]
                ground_tile_distribution += frame_tiles/np.sum(frame_tiles)
                watched_tiles = [[(pitch_idx, yaw_idx), frame_tiles[pitch_idx][yaw_idx]] for pitch_idx in range(len(frame_tiles)) for yaw_idx in range(len(frame_tiles[pitch_idx])) if np.round(frame_tiles[pitch_idx][yaw_idx], 2) > 0 ]
                # tiles_weight = 1.0/np.sum([tile[1] for tile in watched_tiles])
                # print(watched_tiles)
                tiles_weight = 1.0
                # Find download tiles and watched tiles
                frame_watched_rates = 0
                for t_idx in range(len(watched_tiles)):
                    pitch, yaw = watched_tiles[t_idx][0][0], watched_tiles[t_idx][0][1]
                    tile_a, tile_b, tile_w = self.matrix[0][pitch][0], self.matrix[1][pitch][0], self.matrix[2][pitch][0]
                    # print(self.matrix[0], self.matrix[1], self.matrix[2])
                    # print(tile_a, tile_b, tile_w,' wwww')
                    if (pitch, yaw) in seg_tiles:
                        frame_tile_count += 1
                        frame_watched_rates += watched_tiles[t_idx][1]*seg_tiles[(pitch, yaw)]*Config.tile_ratio
                        ## FOr TMM calculate quality
                        ## 1st get log function a, b
                        
                        tile_data_size = get_tile_data_size_in_byte(seg_tiles[(pitch, yaw)], n_frames)
                        # print(tile_data_size)
                        tile_quality = tile_a + tile_b*np.log(tile_data_size)
                        # print(tile_quality)
                        viewed_ratio = watched_tiles[t_idx][1]
                        # print(viewed_ratio, 'ratio')
                        weight = viewed_ratio*tile_w
                        frame_quality += weight*tile_quality
                        frame_view_weight += weight
                    else:
                        # print("not downloaded for ", pitch, yaw)
                        #should still add weight to quality
                        viewed_ratio = watched_tiles[t_idx][1]
                        # print(viewed_ratio, 'ratio and ', tile_w)
                        weight = viewed_ratio*tile_w
                        frame_view_weight += weight
                    # print("curr frame total tweight ", frame_view_weight)
                        # print("view weight", frame_view_weight)
                # FOr TMM calculate frame level weighted psnr
                if frame_tile_count == 0:
                    print('seg ', seg_idx, ' frame ', f_idx, ' is totally wrong')
                    print(frame_watched_rates, frame_quality)
                # print(frame_quality, 'total quality')
                # print(frame_view_weight, 'weight')
                counter += frame_tile_count
                frame_weighted_PSNR = frame_quality/frame_view_weight
                # print('frame ', f_idx, ' quality ', frame_weighted_PSNR)
                # Then add to segment level
                seg_quality += frame_weighted_PSNR

                normalized_frame_rate = tiles_weight * frame_watched_rates
                seg_rates += normalized_frame_rate
                # print(normalized_frame_rate, 'frame rate')
            #################################################
            ## For TMM
            # Get weighted sum of a frame
            seg_quality *= frame_weight
            counter *= frame_weight
            # print(counter, 'average tiles are watched for seg ', seg_idx)
            viewed_wspsnr.append([seg_idx, seg_quality])

            ## record ratios
            ####################################
            ground_tile_distribution *= frame_weight
            seg_rates *= frame_weight
            # print(seg_quality, seg_rates, 'compare')
            # if len(self.cross_lists) and seg_idx == self.cross_lists[0][0]:
            if seg_idx != last_timeout_seg:
                if SHOW_ENTRO:
                    # print("seg id: ", seg_idx)
                    # print("GT dis: ", ground_tile_distribution)
                    if seg_idx in self.cross_lists:
                        print("Predicted: ", self.cross_lists[seg_idx])
                    else:
                        print("Predicted does not exist.")
                if seg_idx in self.cross_lists:
                    [pred_dis, gap_in_s, p_type] = self.cross_lists[seg_idx]
                    # print(ground_tile_distribution, self.cross_lists[seg_idx])
                    cross_entropy = tile_cross_entropy(ground_tile_distribution, pred_dis)
                    corrcoe = correlation_coefficient(ground_tile_distribution, pred_dis)
                    cross_entropys.append([seg_idx, cross_entropy, corrcoe, gap_in_s, p_type])

                    new_ratio = get_tile_overlap_ratio(ground_tile_distribution, pred_dis) 

                    # Get normalized prediction (zero mean, unit std)
                    normalized = (pred_dis - np.mean(pred_dis))/np.std(pred_dis)
                    # print(np.max(normalized))
                    # assert np.max(normalized) <= 1
                    nss_score = np.sum(normalized * scanpath)/np.sum(scanpath)
                    # print(nss_score, np.sum(scanpath), total_sp)
                    # assert nss_score <= 1 and nss_score >= -1
                    ## record ratios
                    seg_ratios = calculate_ratios(ground_tile_distribution, pred_dis)
                    viewed_ratios.append([seg_idx, seg_ratios, gap_in_s, p_type])
                else:
                    print("Predicted does not exist at seg: ", seg_idx)
                    nss_score = 0
                    new_ratio = 0

            video_nss.append([seg_idx, nss_score])
            overlap_ratio.append([seg_idx, new_ratio])
            # del self.cross_lists[0]
            video_rates.append([seg_idx, seg_rates])
            if seg_idx == last_timeout_seg:
                freezing_time += self.freezing_tol/Config.ms_in_s
            video_freeze.append([seg_idx, freezing_time])
        # print(self.u_id, cross_entropys)
        # input()
        self.save_rate(video_rates)
        self.save_freeze(video_freeze)
        self.save_cross_entropy(cross_entropys)
        self.save_buffer_len(buffer_record)
        self.save_download_size(download_sizes)
        self.save_wspsnr(viewed_wspsnr)
        self.save_ratios(viewed_ratios)
        self.save_nss(overlap_ratio)
        self.save_fov()
        return video_rates, video_freeze, cross_entropys, overlap_ratio

    # def save_ratios(self, ratios):
    #     if not os.path.isdir(Config.info_data_path + 'ratios/'):
    #         os.makedirs(Config.info_data_path + 'ratios/')
    #     log_path = Config.info_data_path + 'ratios/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     rate_write = open(log_path, 'w')
    #     for i in range(len(ratios)):
    #         rate_write.write( str(ratios[i][0]) + ' ' + str(ratios[i][1])  + ' ' + str(ratios[i][2]) + ' ' + str(ratios[i][3]))
    #         rate_write.write('\n')

    # def save_freeze(self, freeze):
    #     if not os.path.isdir(Config.info_data_path + 'freeze/'):
    #         os.makedirs(Config.info_data_path + 'freeze/')
    #     log_path = Config.info_data_path + 'freeze/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     freeze_write = open(log_path, 'w')
    #     for i in range(len(freeze)):
    #         freeze_write.write( str(freeze[i][0]) + ' ' + str(freeze[i][1]))
    #         freeze_write.write('\n')

    # def save_buffer_len(self, buffer_len):
    #     if not os.path.isdir(Config.info_data_path + 'buffer/'):
    #         os.makedirs(Config.info_data_path + 'buffer/')
    #     log_path = Config.info_data_path + 'buffer/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     buffer_write = open(log_path, 'w')
    #     for i in range(len(buffer_len)):
    #         buffer_write.write( str(buffer_len[i][0]) + ' ' + str(buffer_len[i][1]))
    #         buffer_write.write('\n')

    # def save_download_size(self, download_size):
    #     if not os.path.isdir(Config.info_data_path + 'size/'):
    #         os.makedirs(Config.info_data_path + 'size/')
    #     log_path = Config.info_data_path + 'size/user' + str(self.u_id) + '_coor' + str(Config.coordinate_fov_prediction) + '_latency' + str(Config.latency_optimization) + '.txt'
    #     size_write = open(log_path, 'w')
    #     for i in range(len(download_size)):
    #         size_write.write( str(download_size[i][0]) + ' ' + str(download_size[i][1]))
    #         size_write.write('\n')

    # def evaluate_performance(self):
    #     # Info is stored in self.downloaded_tiles
    #     # Save info first of all
    #     # self.save_info()
    #     seg_idx = None
    #     last_timeout_seg = None
    #     video_rates = []
    #     video_freeze = []
    #     cross_entropys = []
    #     download_sizes = []
    #     viewed_wspsnr = []
    #     viewed_ratios = []

    #     buffer_initial_time = 0.0
    #     buffer_record = [(buffer_initial_time, 0.0)]
    #     print('len of cross list: ', len(self.cross_lists))
    #     # print(self.downloaded_tiles)
    #     for seg_info in self.downloaded_tiles:
    #         seg_idx = seg_info[0]
    #         download_info = seg_info[1]
    #         rates = seg_info[2]
    #         download_time = download_info[0] + np.sum([tile[3] for tile in download_info[1:]])
    #         download_size = np.sum([tile[2] for tile in download_info[1:]])
    #         buffer_initial_time += download_time
    #         current_buffer_len = download_info[-1][-2]
    #         buffer_record.append((buffer_initial_time, current_buffer_len))
    #         freezing_time = np.round(np.sum([tile[4] for tile in download_info[1:]])/Config.ms_in_s, 2)
    #         if freezing_time == self.freezing_tol/Config.ms_in_s:
    #             last_timeout_seg = seg_idx
    #             continue
    #         download_sizes.append((seg_idx, download_size))
    #         seg_tiles = {}
    #         for i in range(1, len(download_info)):
    #             # Translate [(pitch, yaw), 1] to float pitch.yaw
    #             # assert download_info[i][0][0][0]+download_info[i][0][0][1]*0.1 not in seg_tiles
    #             seg_tiles[(download_info[i][0][0][0], download_info[i][0][0][1])] = rates[i-1]
    #         # print(len(download_info), ' tiles are downloaded for seg ', seg_idx)
    #         # print(seg_tiles)
    #         frame_info = self.fov_trace[seg_idx]
    #         # assert frame_info[0] == seg_idx
    #         n_frames = len(frame_info)-1
    #         # print('number of frames, ', n_frames)
    #         seg_rates = 0.0
    #         frame_weight = 1.0/n_frames 

    #         ### FOR TMM
    #         seg_quality = 0
    #         ground_tile_distribution = np.zeros((Config.n_pitch, Config.n_yaw))
    #         counter = 0

    #         # for ratios
    #         # seg_ratios = []
    #         for f_idx in range(1, len(frame_info)):
    #             # For TMM to calculate frame psnr
    #             frame_tile_count = 0
    #             frame_quality = 0
    #             frame_view_weight = 0
    #             frame_tiles = self.tile_map[int(frame_info[f_idx][1][1]/np.pi*180.0+90.0)][int(frame_info[f_idx][1][0]/np.pi*180.0+180.0)]

    #             # calculate ratios
    #             # if seg_idx in self.cross_lists:
    #             #     frame_ratio = np.sum(self.cross_lists[seg_idx][:]*frame_tiles)
    #             # else:
    #             #     frame_ratio = 0
    #             # seg_ratios += [frame_ratio]
    #             ground_tile_distribution += frame_tiles/np.sum(frame_tiles)
    #             watched_tiles = [[(pitch_idx, yaw_idx), frame_tiles[pitch_idx][yaw_idx]] for pitch_idx in range(len(frame_tiles)) for yaw_idx in range(len(frame_tiles[pitch_idx])) if np.round(frame_tiles[pitch_idx][yaw_idx], 2) > 0 ]
    #             # tiles_weight = 1.0/np.sum([tile[1] for tile in watched_tiles])
    #             # print(watched_tiles)
    #             tiles_weight = 1.0
    #             # Find download tiles and watched tiles
    #             frame_watched_rates = 0
    #             for t_idx in range(len(watched_tiles)):
    #                 pitch, yaw = watched_tiles[t_idx][0][0], watched_tiles[t_idx][0][1]
    #                 tile_a, tile_b, tile_w = self.matrix[0][pitch][0], self.matrix[1][pitch][0], self.matrix[2][pitch][0]
    #                 # print(self.matrix[0], self.matrix[1], self.matrix[2])
    #                 # print(tile_a, tile_b, tile_w,' wwww')
    #                 if (pitch, yaw) in seg_tiles:
    #                     frame_tile_count += 1
    #                     frame_watched_rates += watched_tiles[t_idx][1]*seg_tiles[(pitch, yaw)]*Config.tile_ratio
    #                     ## FOr TMM calculate quality
    #                     ## 1st get log function a, b
                        
    #                     tile_data_size = get_tile_data_size_in_byte(seg_tiles[(pitch, yaw)], n_frames)
    #                     # print(tile_data_size)
    #                     tile_quality = tile_a + tile_b*np.log(tile_data_size)
    #                     # print(tile_quality)
    #                     viewed_ratio = watched_tiles[t_idx][1]
    #                     # print(viewed_ratio, 'ratio')
    #                     weight = viewed_ratio*tile_w
    #                     frame_quality += weight*tile_quality
    #                     frame_view_weight += weight
    #                 else:
    #                     # print("not downloaded for ", pitch, yaw)
    #                     #should still add weight to quality
    #                     viewed_ratio = watched_tiles[t_idx][1]
    #                     # print(viewed_ratio, 'ratio and ', tile_w)
    #                     weight = viewed_ratio*tile_w
    #                     frame_view_weight += weight
    #                 # print("curr frame total tweight ", frame_view_weight)
    #                     # print("view weight", frame_view_weight)
    #             # FOr TMM calculate frame level weighted psnr
    #             if frame_tile_count == 0:
    #                 print('seg ', seg_idx, ' frame ', f_idx, ' is totally wrong')
    #                 print(frame_watched_rates, frame_quality)
    #             # print(frame_quality, 'total quality')
    #             # print(frame_view_weight, 'weight')
    #             counter += frame_tile_count
    #             frame_weighted_PSNR = frame_quality/frame_view_weight
    #             # print('frame ', f_idx, ' quality ', frame_weighted_PSNR)
    #             # Then add to segment level
    #             seg_quality += frame_weighted_PSNR

    #             normalized_frame_rate = tiles_weight * frame_watched_rates
    #             seg_rates += normalized_frame_rate
    #             # print(normalized_frame_rate, 'frame rate')
    #         #################################################
    #         ## For TMM
    #         # Get weighted sum of a frame
    #         seg_quality *= frame_weight
    #         counter *= frame_weight
    #         # print(counter, 'average tiles are watched for seg ', seg_idx)
    #         viewed_wspsnr.append([seg_idx, seg_quality])

    #         ## record ratios
    #         ####################################
    #         ground_tile_distribution *= frame_weight
    #         seg_rates *= frame_weight
    #         # print(seg_quality, seg_rates, 'compare')
    #         # if len(self.cross_lists) and seg_idx == self.cross_lists[0][0]:
    #         if seg_idx != last_timeout_seg:
    #             if SHOW_ENTRO:
    #                 # print("seg id: ", seg_idx)
    #                 # print("GT dis: ", ground_tile_distribution)
    #                 if seg_idx in self.cross_lists:
    #                     print("Predicted: ", self.cross_lists[seg_idx])
    #                 else:
    #                     print("Predicted does not exist.")
    #             if seg_idx in self.cross_lists:
    #                 [pred_dis, gap_in_s, p_type] = self.cross_lists[seg_idx]
    #                 # print(ground_tile_distribution, self.cross_lists[seg_idx])
    #                 cross_entropy = tile_cross_entropy(ground_tile_distribution, pred_dis)
    #                 corrcoe = correlation_coefficient(ground_tile_distribution, pred_dis)
    #                 cross_entropys.append([seg_idx, cross_entropy, corrcoe, gap_in_s, p_type])

    #                 ## record ratios
    #                 seg_ratios = calculate_ratios(ground_tile_distribution, pred_dis)
    #                 viewed_ratios.append([seg_idx, seg_ratios, gap_in_s, p_type])
    #             else:
    #                 print("Predicted does not exist at seg: ", seg_idx)
    #         # del self.cross_lists[0]
    #         video_rates.append([seg_idx, seg_rates])
    #         if seg_idx == last_timeout_seg:
    #             freezing_time += self.freezing_tol/Config.ms_in_s
    #         video_freeze.append([seg_idx, freezing_time])
    #     # print(self.u_id, cross_entropys)
    #     # input()
    #     self.save_rate(video_rates)
    #     self.save_freeze(video_freeze)
    #     self.save_cross_entropy(cross_entropys)
    #     self.save_buffer_len(buffer_record)
    #     self.save_download_size(download_sizes)
    #     self.save_wspsnr(viewed_wspsnr)
    #     self.save_ratios(viewed_ratios)
    #     self.save_fov()
    #     return video_rates, video_freeze, cross_entropys
