class Config(object):
    show_system = 0
    debug = 0                   # For debug
    fov_debug = 0               # For fov debug
    bw_debug = 0                # For bw debug
    choose_tile_debug = 0       # For choosing tile debug
    enable_cache = 0            # Don't consider cache for enhanced work

    # In new project, to control new technics:
    # v0: mmsys: use sim
    # v1: average: all sim = 1, no time weight
    # v2: time: (always used) combination between others' tile distribution and self prediction (decayed by temporal), use sim, use time weight
    # v3: both: v1 and v2, all sim = 1, use time weight
    # V4: dbscan_ave    (equal to v0 or v1)
    # v5: dbsacan + time 
    # v6: DNN using 3 distributions
    # v7: DNN using function to decay heatmap
    # v8: DNN self prediction
    # v9: TLP self
    # v10: COOR without buffer/lat assign
    # V11: DNN coor without buffer 
    # v12: not rate allocation optimize
    new_prediction_version = 11
    #################################################

    # Keeo using s_version = 2 or 1 in new project.
    # DO NOT CHANGE THIS
    
    if new_prediction_version == 9 or new_prediction_version == 8:
        s_version = 0
    elif new_prediction_version == 10 or new_prediction_version == 11:
        s_version = 1
    else:
        s_version = 2       # 0 to 2 (MMSys version control) 0: self prediction; 1: collaborative prediction; 2: latency/buffer upper bound assignmen
        
    if s_version == 0:
        latency_optimization = 0    # Whether control initial latency based on bandwidth
        coordinate_fov_prediction = 0   # If using joint tile selection or independent
    elif s_version == 1:
        latency_optimization = 0    # Whether control initial latency based on bandwidth
        coordinate_fov_prediction = 1   # If using joint tile selection or independent
    elif s_version == 2:
        latency_optimization = 1    # Whether control initial latency based on bandwidth
        coordinate_fov_prediction = 1   # If using joint tile selection or independent
    
    weight_average = 1
    represent_update_interval = 5000.0
    show_cluster = 0            # Should DBSCAN clustering
    show_kf = 0                 # Show kalman filter points
    neighbors_show = 0          # Show sililarity of neighbors
    
    video_version = 49           # 0 to 8, different videos      <=================== CHANGE TESTING CASE HERE   3 (train) and 5 (test) are vertically middle
    USE_5G = 0                  # Whether use 5g bw traces, TMM set to '0'.
    kf_predict = 0              # Whether use kf to do prediction

    n_dis_tth = 5
    dbscan_ave_tth = 5
    # Different algorithms control
    bw_prediction_method = 1        # 0: mean, 1: Harmonic mean 2, RLS
    neighbors_weight_method = 0     # 
    indep_rate_allocation_method = 2      # (only happen in independent case) 0: equally assign, 1, proportional, 2 TMM, weighted
    joint_rate_allocation_method = 1       # 0, rate allocation based on attention map; 1: based on weight/b/attention
    if new_prediction_version == 12:
        joint_rate_allocation_method = 0
        
    num_users = 31                   # Number of users 
    randomSeed = 30              
    tsinghua_fov_data_path = '../Formated_Data/Experiment_2/'
    pickle_root_path = '../new_pickled_data/'
    figure_path = './figures/version_' + str(new_prediction_version) + '/'
    download_seq_path = './cache/version_' + str(new_prediction_version) + '/'
    info_data_path = './data/version_' + str(new_prediction_version) + '/'
    if USE_5G:
        ori_bw_trace_path = '../new_mix_format_sample/'
        bw_trace_path = '../filtered_data/'
    else:
        bw_trace_path = '../new_5G/filtered_5g/'
    # tile_map_dir = '../tile_map/fov_map.mat'
    tile_map_dir_new = '../tile_map/fov_map_90.mat'

    qr_map_dir = '../tile_map/qr_1007.mat'
 
    if enable_cache:
        represent_file_path = './represent/version_' + str(new_prediction_version) + '/'

    # Configuration
    ms_in_s = 1000.0
    num_fov_per_seg = 3
    transform_sim = 2               # 0: erp distance, 1: adjsuted yaw distance and pitch distance; 2: directly use distance on sphere
    db_scan_tth = 3
    dbscan_eps = 0.8                # DBSCAN distance tth
    server_fov_pre_len = 3
    predict_tth_len = 15            # At leat 15 frames needed to predict/truncate
    zero_prediction_tth = 15
    trun_regression_order = 1
    distance_tth = 25               # At most 30 frame
    neighbor_dis_tth = 30           # Threshold of distance to define close neighbor, here 50 degree
    neighbors_upperbound = 3        # At most use 5 closest neighbors

    # Server configuration
    seg_duration = 1000.0
    n_yaw = 32               # 360/32
    n_pitch = 16             # 180/16
    tile_ratio = 1.0/(n_yaw*n_pitch)
    initial_latencies = [20.0, 20.0]    # 10s/20s, corresponding with latencies [-1]
    encoding_allocation_version = 0
    fov_update_per_upload = 0           # Whehter update saliency map per upload
    table_delete_interval = 25          # How frequently update saliency map
    bw_his_len = 10
    coordinate_fov_tth = 5

    # Parameters
    p1 = 3.0
    p2 = 0.5
    alpha = 0.15
    
    # Client configuration
    if USE_5G:
        bitrate = [10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0]
        default_rates = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        default_rates_v1 = [100.0, 100.0, 1000.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        latency_control_version = 1
    else:
        bitrate = [3.0, 5.0, 10.0, 20.0, 40.0, 70.0, 100.0]
        default_rates = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        default_rates_v1 = [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
        latency_control_version = 0

    latencies = [[[0.2, 0.3, 0.3, 0.1], [3, 8, 13, 19]],
                 [[0.1, 0.4, 0.4, 0.1], [3, 8, 13, 19]]]
    enhanced_extra_latency = 10.
    # enhanced_rsd = [0.115, 0.279, 0.346, 0.41]    # rsd for entire trace
    enhanced_rsd = [0.014068231051695522, 0.3433499343705754, 0.4812844219599091, 0.7405726035959522]     # for first 10 seconds
    rtt_low = 10.0
    rtt_high = 20.0 
    packet_payload_portion = 0.98
    freezing_tol = 3000.0   
    user_start_up_th = 2000.0
    default_tiles = [[(1,2), 1], [(1,3), 1],
                     [(2,2), 1], [(2,3), 1],
                     [(3,2), 1], [(3,3), 1]]
    default_tiles_v1 = [[(1,2), 1], [(1,3), 1],
                        [(2,2), 1], [(2,3), 1],
                        [(3,2), 1], [(3,3), 1],
                        [(4,2), 0], [(4,3), 0]]
    user_buffer_upper = 3000.0

    bw_traces_group = [set([105, 102, 95, 73, 177, 125, 167, 82, 198, 190, 134, 101]),
                       set([182, 41, 60, 90, 178, 15, 65, 13, 0, 37, 120, 196]), 
                       set([67, 112, 128, 175, 22, 96, 151, 5, 159, 148, 161, 166]), 
                       set([10, 76, 163, 185, 63, 12, 138, 88, 83, 149, 53, 110])]


    ## Configureatin for DNN fov prediction
    target_his_fov_len = 10  # 10 segments
    target_user_pred_len = 5    # 5 segments

    ## FOr mm
    bw_scale = 1.0
    v_length = 200
    num_interval = 30

    first_prediction_interval = 1
