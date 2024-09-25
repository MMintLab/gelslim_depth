import gelslim_depth.main_config as main_config

#TRAINING OPTIONS
weights_name = 'unet_bigdata'
weights_path = 'train_output/weights/'
loss_curve_path = 'train_output/loss_curves/'
dataset_path = main_config.DATA_PATH+'/'
num_images_to_display_live = 5
exclude_objects = []
batch_size = 16
val_loss_SMA_window = 10
training_learning_rate = 0.001
validation_loss_count_threshold = 5
weight_decay = 1e-06
train_indefinitely = True
#device = device(type='cuda', index=1)
save_at_epochs = [200]
plot_every_epoch = 1


#DATA PROCESSING OPTIONS
depth_image_blur_kernel = 1
downsample_factor = 0.5
use_difference_image = True
interp_method = 'area'


#CNN OPTIONS AND PARAMETERS
input_tactile_image_size = (160, 213)
CNN_dimensions = [64, 128, 256, 512, 1024]
upconv_stride = 2
maxpool_size = 2
model_type = 'unet'
activation_func = 'relu'
kernel_size = 3


#NORMALIZATION PARAMETERS
image_normalization_method = '0_255_to_0_1'
image_normalization_parameters = None
depth_normalization_method = 'min_max_to_0_-1'
depth_normalization_parameters = (-1.9180814027786255, 0.0)
norm_scale = 0.9


#OBJECTS
train_objects = ['pattern_05_3_lines_angle_2_train.pt', 'pattern_02_2_lines_angle_2_train.pt', 'peg3_train.pt', 'pattern_32_train.pt', 'pattern_03_2_lines_angle_3_train.pt', 'pattern_36_train.pt', 'pattern_33_train.pt', 'pattern_06_5_lines_angle_1_train.pt', 'peg1_train.pt', 'pattern_31_rod_train.pt']
validation_objects = ['peg2_val.pt', 'pattern_05_3_lines_angle_2_val.pt', 'pattern_02_2_lines_angle_2_val.pt', 'peg3_val.pt', 'pattern_32_val.pt', 'pattern_37_val.pt', 'pattern_03_2_lines_angle_3_val.pt', 'pattern_04_3_lines_angle_1_val.pt', 'pattern_36_val.pt', 'pattern_33_val.pt', 'pattern_06_5_lines_angle_1_val.pt', 'peg1_val.pt', 'pattern_31_rod_val.pt']
test_objects = ['pattern_05_3_lines_angle_2_test.pt', 'pattern_02_2_lines_angle_2_test.pt', 'peg3_test.pt', 'pattern_32_test.pt', 'pattern_03_2_lines_angle_3_test.pt', 'pattern_36_test.pt', 'pattern_33_test.pt', 'pattern_01_2_lines_angle_1_test.pt', 'pattern_06_5_lines_angle_1_test.pt', 'peg1_test.pt', 'pattern_35_test.pt', 'pattern_31_rod_test.pt']
real_train_objects = ['button.pt', 'ping_pong.pt']
real_validation_objects = ['marble.pt', 'edge.pt']
real_test_objects = ['hex_key.pt']