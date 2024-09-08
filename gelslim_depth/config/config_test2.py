weights_name = 'test2'
weights_path = 'train_output/weights/'
loss_curve_path = 'train_output/loss_curves/'
dataset_path = 'data/'
train_objects = ['pattern_31_rod_train.pt', 'pattern_02_2_lines_angle_2_train.pt', 'pattern_32_train.pt', 'pattern_06_5_lines_angle_1_train.pt', 'pattern_33_train.pt', 'pattern_36_train.pt', 'pattern_05_3_lines_angle_2_train.pt', 'pattern_03_2_lines_angle_3_train.pt']
validation_objects = ['pattern_04_3_lines_angle_1_val.pt', 'pattern_37_val.pt']
test_objects = ['pattern_01_2_lines_angle_1_test.pt', 'pattern_35_test.pt']
batch_size = 16
kernel_size = 3
val_loss_SMA_window = 10
training_learning_rate = 0.001
validation_loss_count_threshold = 5
weight_decay = 1e-06
activation_func = 'relu'
latent_dimension = 1024
norm_scale = 0.9
train_indefinitely = True
num_CNN_layers = 3
num_FCpre_layers = 1
CNN_dimensions = [32, 64, 128]
FCpre_dimensions = [1024]
downsample_factor = 0.5
maxpool_size = 2
model_type = 'vae_transform'
input_tactile_image_size = (160, 213)
depth_image_blur_kernel = 1
#device = device(type='cuda', index=2)
KL_weight = 0.001
depth_normalization_parameters = (tensor(-0.9994), tensor(0.))
