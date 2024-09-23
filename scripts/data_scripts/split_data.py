import torch
import os
from tqdm import tqdm
import sys
import gelslim_depth.main_config as main_config

#WARNING: This script will delete the original data files after splitting them into train, validation, and test sets, uncomment the line that deletes the original file if you want to keep the original files

#this script will split all the data files in the data directory into train, validation, and test sets, all keys in the data files will be split in the same way, so they should already be aligned
#usage: python3 scripts/data_scripts/split_data.py <device>
#example1: python3 scripts/data_scripts/split_data.py cuda:0
#example2: python3 scripts/data_scripts/split_data.py cpu

data_dir = main_config.DATA_PATH
data_files = os.listdir(data_dir)
#remove files that are not .pt files
data_files = [f for f in data_files if f[-3:] == '.pt']

device = sys.argv[1]

#set the split ratios
train_val_test_split = [0.8, 0.1, 0.1]

num_objects = len(data_files)

for i in tqdm(range(num_objects)):
    filename = data_files[i]
    data = torch.load(os.path.join(data_dir, data_files[i]), map_location=device)
    tactile_images = data['tactile_image']
    num_datapoints = tactile_images.size()[0]
    #shuffle the data
    indices = torch.randperm(num_datapoints)
    train_index = int(num_datapoints*train_val_test_split[0])
    val_index = int(num_datapoints*(train_val_test_split[0] + train_val_test_split[1]))
    tactile_images = tactile_images[indices, ...]
    train_tactile_images = tactile_images[:train_index, ...]
    val_tactile_images = tactile_images[train_index:val_index, ...]
    test_tactile_images = tactile_images[val_index:, ...]
    train_dict = {'tactile_image': train_tactile_images}
    val_dict = {'tactile_image': val_tactile_images}
    test_dict = {'tactile_image': test_tactile_images}
    if 'depth_image' in data.keys():
        depth_images = data['depth_image']
        depth_images = depth_images[indices, ...]
        train_depth_images = depth_images[:train_index, ...]
        val_depth_images = depth_images[train_index:val_index, ...]
        test_depth_images = depth_images[val_index:, ...]
        train_dict['depth_image'] = train_depth_images
        val_dict['depth_image'] = val_depth_images
        test_dict['depth_image'] = test_depth_images
    else:
        print('[INFO] No depth images found in the data file'+filename+'. Still splitting the data. You can generate depth images using the depth_generation.py script')
    if 'in_hand_pose' in data.keys():
        in_hand_pose = data['in_hand_pose']
        in_hand_pose = in_hand_pose[indices, ...]
        train_in_hand_pose = in_hand_pose[:train_index, ...]
        val_in_hand_pose = in_hand_pose[train_index:val_index, ...]
        test_in_hand_pose = in_hand_pose[val_index:, ...]
        train_dict['in_hand_pose'] = train_in_hand_pose
        val_dict['in_hand_pose'] = val_in_hand_pose
        test_dict['in_hand_pose'] = test_in_hand_pose
    else:
        print('[INFO] No in hand poses found in the data file'+filename+'. Still splitting the data. In hand poses are necessary to generate the "ground truth" depth images from mesh files.')
    if 'base_tactile_image' in data.keys():
        base_tactile_images = data['base_tactile_image']
        base_tactile_images = base_tactile_images[indices, ...]
        train_base_tactile_images = base_tactile_images[:train_index, ...]
        val_base_tactile_images = base_tactile_images[train_index:val_index, ...]
        test_base_tactile_images = base_tactile_images[val_index:, ...]
        train_dict['base_tactile_image'] = train_base_tactile_images
        val_dict['base_tactile_image'] = val_base_tactile_images
        test_dict['base_tactile_image'] = test_base_tactile_images
    else:
        print('[INFO] No base tactile images found in the data file'+filename+'. Still splitting the data. Base (undeformed) tactile images are necessary to generate the difference images.')
    
    train_filename = filename[:-3] + '_train.pt'
    val_filename = filename[:-3] + '_val.pt'
    test_filename = filename[:-3] + '_test.pt'

    #delete the original file, you can comment this out if you want to keep the original file
    os.remove(os.path.join(data_dir, filename))
    
    torch.save(train_dict, os.path.join(data_dir, 'train_data', train_filename))
    torch.save(val_dict, os.path.join(data_dir, 'validation_data', val_filename))
    torch.save(test_dict, os.path.join(data_dir, 'test_data', test_filename))
