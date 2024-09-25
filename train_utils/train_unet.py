import torch
import numpy as np
from gelslim_depth.datasets.general_dataset import GeneralDataset
from gelslim_depth.models.unet import UNet
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Dict
import time
import matplotlib.pyplot as plt
from torch_ema import ExponentialMovingAverage
import os
import argparse
import gelslim_depth.main_config as main_config

#CNN and training parameters
downsample_factor = 0.5

interp_method = 'area'

#CNN_dimensions = [64, 256, 1024]
CNN_dimensions = [64,128,256,512,1024]

kernel_size = 3
maxpool_size = 2

upconv_stride = 2

model_type = 'unet'

val_loss_SMA_window = 10

training_learning_rate = 1e-3

weight_decay = 1e-6

validation_loss_count_threshold = 5

norm_scale = 0.9

batch_size = 16

depth_image_blur_kernel = 1

#configuration for live display
num_images_to_display_live = 5

#normalization settings, the resulting parameters will be saved in the config file
depth_normalization_method = 'min_max_to_0_-1'
image_normalization_method = '0_255_to_0_1'

def MSE_loss(input: Tensor, target: Tensor) -> Tensor:
    return torch.mean((input - target)**2)

def parse_args():
    parser = argparse.ArgumentParser(description='Train an in hand pose estimation model.')
    
    parser.add_argument('weights_name', type=str, help='Name for the weights')
    parser.add_argument('gpu', type=str, help='GPU id to use')
    parser.add_argument('--exclude_objects', nargs='+', help='List of objects to exclude')
    parser.add_argument('--activation_func', type=str, default='relu', choices=['relu', 'tanh', 'mish'], help='Activation function to use')
    parser.add_argument('--train_indefinitely', action='store_true', help='Train past early stopping')
    parser.add_argument('--use_difference_image', action='store_true', help='Use difference tactile image')
    parser.add_argument('--max_datapoints_per_object', type=int, help='Maximum number of datapoints per object')

    return parser.parse_args()

#enter a number to limit the number of objects in each of the train, validation, and test sets, useful for debugging
limit_object_lists = 1

#example shell command, run from the repository root directory:
#python3 train_utils/train_unet.py unet_weights_1 1 --exclude_objects peg1 --activation_func mish --train_indefinitely --use_difference_image --max_datapoints_per_object 1000

##begin configuration

args = parse_args()

weights_name = args.weights_name
#add name of starting weights pth file if you want to fine tune from a previous model
starting_weights = None
gpu = args.gpu
exclude_objects = args.exclude_objects
activation_func = args.activation_func
train_indefinitely = args.train_indefinitely
use_difference_image = args.use_difference_image
max_datapoints_per_object = args.max_datapoints_per_object
save_at_epochs = [200]
last_save_epoch = 0
plot_every_epoch = 1
last_plot_epoch = 0

weights_path = 'train_output/weights/'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

loss_curve_path = 'train_output/loss_curves/'
if not os.path.exists(loss_curve_path):
    os.makedirs(loss_curve_path)

loss_values_path = 'train_output/loss_values/'
if not os.path.exists(loss_values_path):
    os.makedirs(loss_values_path)
loss_values_path += weights_name + '.txt'

live_display_path = 'train_output/live_display/'
if not os.path.exists(live_display_path):
    os.makedirs(live_display_path)

dataset_path = main_config.DATA_PATH+'/'

train_objects = os.listdir(dataset_path+'train_data/')
#replace _train.pt with .pt
train_objects = [f.replace('_train.pt', '.pt') for f in train_objects]

#get list of validation objects from validation_objects.txt
validation_objects_file = dataset_path+'validation_objects.txt'
if os.path.exists(validation_objects_file):
    with open(validation_objects_file, 'r') as f:
        validation_objects_to_exclude = f.read().splitlines()
    #add .pt to the end of the object names
    validation_objects_to_exclude = [f+'.pt' for f in validation_objects_to_exclude]
else:
    validation_objects_to_exclude = []

test_objects_file = dataset_path+'test_objects.txt'
if os.path.exists(test_objects_file):
    with open(test_objects_file, 'r') as f:
        test_objects_to_exclude = f.read().splitlines()
    #add .pt to the end of the object names
    test_objects_to_exclude = [f+'.pt' for f in test_objects_to_exclude]
else:
    test_objects_to_exclude = []

validation_objects = train_objects
test_objects = train_objects

#remove test and validation objects from train_objects
train_objects = [f for f in train_objects if f not in validation_objects_to_exclude and f not in test_objects_to_exclude]
#remove test objects from validation_objects
validation_objects = [f for f in validation_objects if f not in test_objects_to_exclude]
#remove validation objects from test_objects
test_objects = [f for f in test_objects if f not in validation_objects_to_exclude]

if exclude_objects is not None:
    for object_name in exclude_objects:
        for train_object in train_objects:
            if object_name in train_object:
                train_objects.remove(train_object)
        for validation_object in validation_objects:
            if object_name in validation_object:
                validation_objects.remove(validation_object)
        for test_object in test_objects:
            if object_name in test_object:
                test_objects.remove(test_object)
else:
    exclude_objects = []
#replace .pt with _train.pt in train_objects
train_objects = [f[:-3]+'_train.pt' for f in train_objects]

#replace .pt with _val.pt in validation_objects
validation_objects = [f[:-3]+'_val.pt' for f in validation_objects]

#replace .pt with _test.pt in test_objects
test_objects = [f[:-3]+'_test.pt' for f in test_objects]

real_train_objects_file = dataset_path+'real_data/train_real_objects.txt'
if os.path.exists(real_train_objects_file):
    with open(real_train_objects_file, 'r') as f:
        real_train_objects = f.read().splitlines()
    #add .pt to the end of the object names
    real_train_objects = [f+'.pt' for f in real_train_objects]

real_validation_objects_file = dataset_path+'real_data/validation_real_objects.txt'
if os.path.exists(real_validation_objects_file):
    with open(real_validation_objects_file, 'r') as f:
        real_validation_objects = f.read().splitlines()
    #add .pt to the end of the object names
    real_validation_objects = [f+'.pt' for f in real_validation_objects]

real_test_objects_file = dataset_path+'real_data/test_real_objects.txt'
if os.path.exists(real_test_objects_file):
    with open(real_test_objects_file, 'r') as f:
        real_test_objects = f.read().splitlines()
    #add .pt to the end of the object names
    real_test_objects = [f+'.pt' for f in real_test_objects]

if limit_object_lists:
    train_objects = train_objects[:limit_object_lists]
    validation_objects = validation_objects[:limit_object_lists]
    test_objects = test_objects[:limit_object_lists]
    real_train_objects = real_train_objects[:limit_object_lists]
    real_validation_objects = real_validation_objects[:limit_object_lists]
    real_test_objects = real_test_objects[:limit_object_lists]

device = torch.device('cuda:'+gpu if torch.cuda.is_available() else 'cpu')

start_data_load_time = time.time()

#initialize the dataset
TrainDataset = GeneralDataset(directory=dataset_path+'train_data/', pt_file_list=train_objects, extra_directory=dataset_path+'real_data/', extra_pt_list=real_train_objects, use_difference_image=use_difference_image,
                            image_normalization_parameters=None, depth_normalization_method=depth_normalization_method, image_normalization_method=image_normalization_method,
                            separate_fingers=True, downsample_factor=downsample_factor, depth_image_blur_kernel=depth_image_blur_kernel, depth_normalization_parameters = None, norm_scale=norm_scale, max_datapoints_per_object=max_datapoints_per_object,
                            device=device, interp_method=interp_method)

print("Found {} training points".format(len(TrainDataset)))

end_data_load_time = time.time()
print('Training Data Load Time: {}s'.format(end_data_load_time-start_data_load_time))
depth_normalization_parameters = TrainDataset.depth_normalization_parameters
image_normalization_parameters = TrainDataset.image_normalization_parameters
input_tactile_image_size = TrainDataset.input_tactile_image_size

#initialize the validation dataset
ValDataset = GeneralDataset(directory=dataset_path+'validation_data/', pt_file_list=validation_objects, extra_directory=dataset_path+'real_data/', extra_pt_list=real_validation_objects, use_difference_image=use_difference_image,
                            image_normalization_parameters=image_normalization_parameters, depth_normalization_method=depth_normalization_method, image_normalization_method=image_normalization_method,
                            separate_fingers=True, downsample_factor=downsample_factor, depth_image_blur_kernel=depth_image_blur_kernel, depth_normalization_parameters = depth_normalization_parameters, norm_scale=norm_scale, max_datapoints_per_object=max_datapoints_per_object,
                            device=device, interp_method=interp_method)

print("Found {} validation points".format(len(ValDataset)))

#initialize the test dataset
TestDataset = GeneralDataset(directory=dataset_path+'test_data/', pt_file_list=test_objects, extra_directory=dataset_path+'real_data/', extra_pt_list=real_test_objects, use_difference_image=use_difference_image, image_normalization_parameters=image_normalization_parameters,
                             depth_normalization_method=depth_normalization_method, image_normalization_method=image_normalization_method,
                             separate_fingers=True, downsample_factor=downsample_factor, depth_image_blur_kernel=depth_image_blur_kernel, depth_normalization_parameters = depth_normalization_parameters, norm_scale=norm_scale, max_datapoints_per_object=max_datapoints_per_object,
                             device=device, interp_method=interp_method)

print("Found {} test points".format(len(TestDataset)))
      
#initialize the dataloaders
TrainLoader = DataLoader(TrainDataset, batch_size=batch_size, shuffle=True, pin_memory=True)

ValLoader = DataLoader(ValDataset, batch_size=batch_size, shuffle=True, pin_memory=True)

TestLoader = DataLoader(TestDataset, batch_size=batch_size, shuffle=True, pin_memory=True)

unet = UNet(n_channels=3, n_classes=1, layer_dimensions=CNN_dimensions, kernel_size=kernel_size, maxpool_size=maxpool_size, upconv_stride=upconv_stride).to(device)

#initialize the weights

#load the starting weights if they exist, for fine tuning
if starting_weights is not None:
    starting_weights_file = 'train_output/weights/'+starting_weights+'.pth'
    #get absolute path
    starting_weights_file = os.path.expanduser(starting_weights_file)
    unet.load_state_dict(torch.load(starting_weights_file, map_location=device))

#or initialize the weights with a normal distribution
else:
    for name, param in unet.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.01)

#save configuration as a python file
with open('gelslim_depth/config/config_' + weights_name + '.py', 'w') as f:
    f.write('#TRAINING OPTIONS' + '\n')
    f.write('weights_name = ' + repr(weights_name) + '\n')
    f.write('weights_path = ' + repr(weights_path) + '\n')
    f.write('loss_curve_path = ' + repr(loss_curve_path) + '\n')
    f.write('dataset_path = ' + repr(dataset_path) + '\n')
    f.write('num_images_to_display_live = ' + repr(num_images_to_display_live) + '\n')
    f.write('exclude_objects = ' + repr(exclude_objects) + '\n')
    f.write('batch_size = ' + repr(batch_size) + '\n')
    f.write('val_loss_SMA_window = ' + repr(val_loss_SMA_window) + '\n')
    f.write('training_learning_rate = ' + repr(training_learning_rate) + '\n')
    f.write('validation_loss_count_threshold = ' + repr(validation_loss_count_threshold) + '\n')
    f.write('weight_decay = ' + repr(weight_decay) + '\n')
    f.write('train_indefinitely = ' + repr(train_indefinitely) + '\n')
    f.write('#device = ' + repr(device) + '\n')
    f.write('save_at_epochs = ' + repr(save_at_epochs) + '\n')
    f.write('plot_every_epoch = ' + repr(plot_every_epoch) + '\n')

    f.write('\n')
    f.write('#DATA PROCESSING OPTIONS' + '\n')
    f.write('depth_image_blur_kernel = ' + repr(depth_image_blur_kernel) + '\n')
    f.write('downsample_factor = ' + repr(downsample_factor) + '\n')
    f.write('use_difference_image = ' + repr(use_difference_image) + '\n')
    f.write('interp_method = ' + repr(interp_method) + '\n')

    f.write('\n')
    f.write('#CNN OPTIONS AND PARAMETERS' + '\n')
    f.write('input_tactile_image_size = ' + repr(input_tactile_image_size) + '\n')
    f.write('CNN_dimensions = ' + repr(CNN_dimensions) + '\n')
    f.write('upconv_stride = ' + repr(upconv_stride) + '\n')
    f.write('maxpool_size = ' + repr(maxpool_size) + '\n')
    f.write('model_type = ' + repr(model_type) + '\n')
    f.write('activation_func = ' + repr(activation_func) + '\n')
    f.write('kernel_size = ' + repr(kernel_size) + '\n')

    f.write('\n')
    f.write('#NORMALIZATION PARAMETERS' + '\n')
    f.write('image_normalization_method = ' + repr(image_normalization_method) + '\n')
    f.write('image_normalization_parameters = ' + repr(image_normalization_parameters) + '\n')
    f.write('depth_normalization_method = ' + repr(depth_normalization_method) + '\n')
    f.write('depth_normalization_parameters = ' + repr(depth_normalization_parameters) + '\n')
    f.write('norm_scale = ' + repr(norm_scale) + '\n')

    f.write('\n')
    f.write('#OBJECTS' + '\n')
    f.write('train_objects = ' + repr(train_objects) + '\n')
    f.write('validation_objects = ' + repr(validation_objects) + '\n')
    f.write('test_objects = ' + repr(test_objects) + '\n')
    f.write('real_train_objects = ' + repr(real_train_objects) + '\n')
    f.write('real_validation_objects = ' + repr(real_validation_objects) + '\n')
    f.write('real_test_objects = ' + repr(real_test_objects) + '\n')

#initialize adam optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=training_learning_rate, weight_decay=weight_decay)

#initialize exponential moving average for the weights
ema = ExponentialMovingAverage(unet.parameters(), decay=0.995)

H = {"train_loss": [], "validation_loss": [], "test_loss": []}

print('Training the unet')
startTime = time.time()
#validation loss smoothing
validation_loss_increasing = False
prev_validation_loss = 0
prev_unsmoothed_validation_loss = 0
e = 0
validation_loss_upward_counter = 0
validation_array = np.zeros(val_loss_SMA_window)
smoothed_validation_loss = np.mean(validation_array)
min_validation_loss = 1000000
with open(loss_values_path, 'a') as loss_file:
    fig_size = (10, 10)
    train_fig, train_ax = plt.subplots(num_images_to_display_live, 3, figsize=fig_size)
    validation_fig, validation_ax = plt.subplots(num_images_to_display_live, 3, figsize=fig_size)
    test_fig, test_ax = plt.subplots(num_images_to_display_live, 3, figsize=fig_size)

    while not validation_loss_increasing:
        if e - last_plot_epoch >= plot_every_epoch:
            last_plot_epoch = e
            plot = True
        else:
            plot = False
        pre_epoch_time = time.time()
        unet.train()
        train_loss = 0
        num_train_images_selected = 0
        for i, data in enumerate(TrainLoader):
            input_image = data['tactile_image']
            output_target = data['depth_image']
            object_index = data['object_index']
            input_image = input_image.to(device, non_blocking=True)
            output_target = output_target.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = unet(x=input_image)
            if num_train_images_selected < num_images_to_display_live and plot:
                #randomly select yes or no to save
                save_image = np.random.choice([True, False])
                if save_image:
                    image_index = np.random.randint(batch_size)
                    train_ax[num_train_images_selected, 0].imshow((255.0*input_image[image_index, ...].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8))
                    train_ax[num_train_images_selected, 1].imshow(output[image_index, 0, ...].cpu().detach().numpy())
                    train_ax[num_train_images_selected, 2].imshow(output_target[image_index, 0, ...].cpu().detach().numpy())
                    if num_train_images_selected == 0:
                        train_ax[num_train_images_selected, 0].set_title('Input')
                        train_ax[num_train_images_selected, 1].set_title('Output')
                        train_ax[num_train_images_selected, 2].set_title('Ground Truth')
                    train_ax[num_train_images_selected, 0].set_xticks([])
                    train_ax[num_train_images_selected, 1].set_xticks([])
                    train_ax[num_train_images_selected, 2].set_xticks([])
                    train_ax[num_train_images_selected, 0].set_yticks([])
                    train_ax[num_train_images_selected, 1].set_yticks([])
                    train_ax[num_train_images_selected, 2].set_yticks([])
                    num_train_images_selected += 1
            if num_train_images_selected == num_images_to_display_live and plot:
                train_fig.suptitle('Epoch ' + str(e+1) + ' Train Images')
                train_fig.savefig(live_display_path + weights_name +'_train_images.png')
            pred_loss = MSE_loss(input=output, target=output_target)
            if pred_loss.isnan():
                pred_loss = torch.tensor(0.0).to(device)
            loss = pred_loss
            loss.backward()
            optimizer.step()
            ema.update()
            train_loss += loss.item()
        train_loss /= len(TrainLoader)
        H["train_loss"].append(train_loss)
        unet.eval()
        validation_loss = 0
        num_val_images_selected = 0
        for i, data in enumerate(ValLoader):
            input_image = data['tactile_image']
            output_target = data['depth_image']
            object_index = data['object_index']
            input_image = input_image.to(device, non_blocking=True)
            output_target = output_target.to(device, non_blocking=True)
            with ema.average_parameters():
                output = unet(x=input_image)
                if num_val_images_selected < num_images_to_display_live and plot:
                    #randomly select yes or no to save
                    save_image = np.random.choice([True, False])
                    if save_image:
                        image_index = np.random.randint(batch_size)
                        validation_ax[num_val_images_selected, 0].imshow((255.0*input_image[image_index, ...].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8))
                        validation_ax[num_val_images_selected, 1].imshow(output[image_index, 0, ...].cpu().detach().numpy())
                        validation_ax[num_val_images_selected, 2].imshow(output_target[image_index, 0, ...].cpu().detach().numpy())
                        if num_val_images_selected == 0:
                            validation_ax[num_val_images_selected, 0].set_title('Input')
                            validation_ax[num_val_images_selected, 1].set_title('Output')
                            validation_ax[num_val_images_selected, 2].set_title('Ground Truth')
                        validation_ax[num_val_images_selected, 0].set_xticks([])
                        validation_ax[num_val_images_selected, 1].set_xticks([])
                        validation_ax[num_val_images_selected, 2].set_xticks([])
                        validation_ax[num_val_images_selected, 0].set_yticks([])
                        validation_ax[num_val_images_selected, 1].set_yticks([])
                        validation_ax[num_val_images_selected, 2].set_yticks([])
                        num_val_images_selected += 1
                if num_val_images_selected == num_images_to_display_live and plot:
                    validation_fig.suptitle('Epoch ' + str(e+1) + ' Validation Images')
                    validation_fig.savefig(live_display_path + weights_name +'_validation_images.png')
                pred_loss = MSE_loss(input=output, target=output_target)
                if pred_loss.isnan():
                    pred_loss = torch.tensor(0.0).to(device)
                loss = pred_loss
                validation_loss += loss.item()
        validation_loss /= len(ValLoader)
        H["validation_loss"].append(validation_loss)
        test_loss = 0
        num_test_images_selected = 0
        for i, data in enumerate(TestLoader):
            input_image = data['tactile_image']
            output_target = data['depth_image']
            object_index = data['object_index']
            input_image = input_image.to(device, non_blocking=True)
            output_target = output_target.to(device, non_blocking=True)
            with ema.average_parameters():
                output = unet(x=input_image)
                if num_test_images_selected < num_images_to_display_live and plot:
                    #randomly select yes or no to save
                    save_image = np.random.choice([True, False])
                    if save_image:
                        image_index = np.random.randint(batch_size)
                        test_ax[num_test_images_selected, 0].imshow((255.0*input_image[image_index, ...].cpu().detach().numpy().transpose(1, 2, 0)).astype(np.uint8))
                        test_ax[num_test_images_selected, 1].imshow(output[image_index, 0, ...].cpu().detach().numpy())
                        test_ax[num_test_images_selected, 2].imshow(output_target[image_index, 0, ...].cpu().detach().numpy())
                        if num_test_images_selected == 0:
                            test_ax[num_test_images_selected, 0].set_title('Input')
                            test_ax[num_test_images_selected, 1].set_title('Output')
                            test_ax[num_test_images_selected, 2].set_title('Ground Truth')
                        test_ax[num_test_images_selected, 0].set_xticks([])
                        test_ax[num_test_images_selected, 1].set_xticks([])
                        test_ax[num_test_images_selected, 2].set_xticks([])
                        test_ax[num_test_images_selected, 0].set_yticks([])
                        test_ax[num_test_images_selected, 1].set_yticks([])
                        test_ax[num_test_images_selected, 2].set_yticks([])
                        num_test_images_selected += 1
                if num_test_images_selected == num_images_to_display_live and plot:
                    test_fig.suptitle('Epoch ' + str(e+1) + ' Test Images')
                    test_fig.savefig(live_display_path + weights_name +'_test_images.png')
                pred_loss = MSE_loss(input=output, target=output_target)
                if pred_loss.isnan():
                    pred_loss = torch.tensor(0.0).to(device)
                loss = pred_loss
                test_loss += loss.item()
        test_loss /= len(TestLoader)
        H["test_loss"].append(test_loss)
        #append the validation loss to the array
        validation_array[e%val_loss_SMA_window] = validation_loss
        #update the smoothed validation loss
        smoothed_validation_loss = np.mean(validation_array)
        #check if the validation loss is increasing only if 75 epochs have passed
        if True:
            if smoothed_validation_loss > prev_validation_loss:
                validation_loss_upward_counter += 1
            else:
                validation_loss_upward_counter = 0
            if validation_loss_upward_counter > validation_loss_count_threshold:
                validation_loss_increasing = True
                if train_indefinitely:
                    print('Validation loss stopped decreasing at epoch ', e+1)
                    loss_file.write(f'Validation loss stopped decreasing at epoch {e+1}\n')
                    validation_loss_increasing = False
            prev_validation_loss = smoothed_validation_loss
            if validation_loss < min_validation_loss:
                #save the model
                print('Validation loss is at a minimum. Saving the model')
                loss_file.write('Validation loss is at a minimum. Saving the model\n')
                with ema.average_parameters():
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path)
                    torch.save(unet.state_dict(), weights_path + weights_name + '.pth')
                min_validation_loss = validation_loss
        if train_indefinitely and len(save_at_epochs) > 0:
            if e in save_at_epochs:
                with ema.average_parameters():
                    if not os.path.exists(weights_path):
                        os.makedirs(weights_path)
                    torch.save(unet.state_dict(), weights_path + weights_name + '_epoch' + str(e) + '.pth')
        print("[INFO] EPOCH: {}".format(e + 1))
        loss_file.write("[INFO] EPOCH: {}\n".format(e + 1))
        print("Train loss: {:.6f},  Validation loss: {:.6f}, Test loss: {:.6f}".format(
            train_loss, validation_loss, test_loss))
        loss_file.write("Train loss: {:.6f},  Validation loss: {:.6f}, Test loss: {:.6f}\n".format(
            train_loss, validation_loss, test_loss))
        print(f'Time for epoch: {time.time() - pre_epoch_time}')
        loss_file.write(f'Time for epoch: {time.time() - pre_epoch_time}\n')
        
        #truncate all loss values to max of 1 for plotting
        #H["train_loss"] = [min(1, x) for x in H["train_loss"]]
        #H["validation_loss"] = [min(1, x) for x in H["validation_loss"]]
        #H["test_loss"] = [min(1, x) for x in H["test_loss"]]
        #plot and save loss curves
        if plot:
            plt.figure()
            plt.style.use("ggplot")
            plt.plot(H["train_loss"], label="train_loss")
            plt.plot(H["validation_loss"], label="validation_loss")
            plt.plot(H["test_loss"], label="test_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend(loc="upper right")
            #log scale plot
            plt.yscale('log')
            plt.savefig(loss_curve_path + weights_name + '.png')
            plt.close()
        e += 1
    endTime = time.time()
    print('Training complete')
    loss_file.write('Training complete\n')
    print('Training time: {}s'.format(endTime-startTime))
    loss_file.write('Training time: {}s\n'.format(endTime-startTime))
