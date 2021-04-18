
# Tile setting:
# Matlab dir
tile_map_dir = '../tile_map/fov_map.mat'
new_tile_map_dir = '../tile_map/fov_map_new.mat'

## FoV dataset
num_video = 9
num_user = 48
tsinghua_fov_data_path = '../../Formated_Data/Experiment_2/'
tsinghua_pickled_data_path = '../pickled_data/'
tsinghua_seg_ave_attention_path = '../pickled_data/all_seg_attention.p'
tsinghua_seg_ave_attention_path_new = '../pickled_data/all_seg_attention_new.p'

# Fov training
batch_size = 1
stride = 1
running_length = 10		# use past 10 seconds
predict_step = 5		# predict future 5 seconds 
data_chunk_stride = 1	
# num_row = 16
# num_col = 32
num_row = 16
num_col = 32
latent_dim = 16			#
conv_kernel_size = 4
dropout_rate = 0.1
stateful_across_batch = False
model_saving_path = './models/'


predict_step_eva = 1
