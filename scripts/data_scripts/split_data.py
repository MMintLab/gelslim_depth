import torch
import os
from tqdm import tqdm

data_dir = 'data'
data_files = os.listdir(data_dir)
#remove files that are not .pt files
data_files = [f for f in data_files if f[-3:] == '.pt']

import pdb; pdb.set_trace()

train_val_test_split = [0.8, 0.1, 0.1]

num_objects = len(data_files)

for i in tqdm(range(num_objects)):
    filename = data_files[i]
    data = torch.load(os.path.join(data_dir, data_files[i]), map_location='cuda:2')
    tactile_images = data['tactile_image']
    depth_images = data['depth_image']
    in_hand_pose = data['in_hand_pose']
    base_tactile_images = data['base_tactile_image']
    num_datapoints = tactile_images.size()[0]
    #shuffle the data
    indices = torch.randperm(num_datapoints)
    tactile_images = tactile_images[indices, ...]
    depth_images = depth_images[indices, ...]
    in_hand_pose = in_hand_pose[indices, ...]
    base_tactile_images = base_tactile_images[indices, ...]
    #split the data
    train_index = int(num_datapoints*train_val_test_split[0])
    val_index = int(num_datapoints*(train_val_test_split[0] + train_val_test_split[1]))
    train_tactile_images = tactile_images[:train_index, ...]
    train_depth_images = depth_images[:train_index, ...]
    train_in_hand_pose = in_hand_pose[:train_index, ...]
    train_base_tactile_images = base_tactile_images[:train_index, ...]
    val_tactile_images = tactile_images[train_index:val_index, ...]
    val_depth_images = depth_images[train_index:val_index, ...]
    val_in_hand_pose = in_hand_pose[train_index:val_index, ...]
    val_base_tactile_images = base_tactile_images[train_index:val_index, ...]
    test_tactile_images = tactile_images[val_index:, ...]
    test_depth_images = depth_images[val_index:, ...]
    test_in_hand_pose = in_hand_pose[val_index:, ...]
    test_base_tactile_images = base_tactile_images[val_index:, ...]
    train_filename = filename[:-3] + '_train.pt'
    val_filename = filename[:-3] + '_val.pt'
    test_filename = filename[:-3] + '_test.pt'
    train_dict = {'tactile_image': train_tactile_images, 'depth_image': train_depth_images, 'in_hand_pose': train_in_hand_pose, 'base_tactile_image': train_base_tactile_images}
    val_dict = {'tactile_image': val_tactile_images, 'depth_image': val_depth_images, 'in_hand_pose': val_in_hand_pose, 'base_tactile_image': val_base_tactile_images}
    test_dict = {'tactile_image': test_tactile_images, 'depth_image': test_depth_images, 'in_hand_pose': test_in_hand_pose, 'base_tactile_image': test_base_tactile_images}
    #delete the original file
    os.remove(os.path.join(data_dir, filename))
    torch.save(train_dict, os.path.join(data_dir, 'train_data', train_filename))
    torch.save(val_dict, os.path.join(data_dir, 'validation_data', val_filename))
    torch.save(test_dict, os.path.join(data_dir, 'test_data', test_filename))
