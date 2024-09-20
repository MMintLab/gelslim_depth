import torch
from typing import Dict

def normalize_tactile_image(tactile_image, image_normalization_method, norm_scale, image_normalization_params=None):
    if '0_255' not in image_normalization_method:
        mins, maxes, means, stds = image_normalization_params
    if image_normalization_method == 'min_max_to_-1_1':
        scale = norm_scale
        bias = 0.5*(torch.tensor(maxes)+torch.tensor(mins)).tolist()
        denominator = (torch.tensor(maxes)-torch.tensor(mins)).tolist()
    elif image_normalization_method == 'mean_std':
        scale = 1.0
        bias = means
        denominator = stds
    elif image_normalization_method == '0_255_to_-1_1':
        scale = 2.0
        bias = [127.5]
        denominator = [255.0]
    elif image_normalization_method == '0_255_to_0_1':
        scale = 1.0
        bias = [0.0]
        denominator = [255.0]
    if len(tactile_image.shape) == 3:
        #no batch dimension
        num_channels = tactile_image.shape[0]
        new_tactile_image = torch.zeros_like(tactile_image)
        for i in range(num_channels):
            new_tactile_image[i,...] = scale*(tactile_image[i,...] - bias[min(i,len(bias)-1)])/denominator[min(i,len(denominator)-1)]
    elif len(tactile_image.shape) == 4:
        #batch dimension
        num_channels = tactile_image.shape[1]
        new_tactile_image = torch.zeros_like(tactile_image)
        for i in range(num_channels):
            new_tactile_image[:,i,...] = scale*(tactile_image[:,i,...] - bias[min(i,len(bias)-1)])/denominator[min(i,len(denominator)-1)]
    return new_tactile_image

def denormalize_tactile_image(tactile_image, image_normalization_method, norm_scale, image_normalization_params=None) -> Dict[str, torch.Tensor]:
    if '0_255' not in image_normalization_method:
        mins, maxes, means, stds = image_normalization_params
    if image_normalization_method == 'min_max_to_-1_1':
        scale = norm_scale
        bias = 0.5*(torch.tensor(maxes)+torch.tensor(mins)).tolist()
        denominator = (torch.tensor(maxes)-torch.tensor(mins)).tolist()
    elif image_normalization_method == 'mean_std':
        scale = 1.0
        bias = means
        denominator = stds
    elif image_normalization_method == '0_255_to_-1_1':
        scale = 2.0
        bias = [127.5]
        denominator = [255.0]
    elif image_normalization_method == '0_255_to_0_1':
        scale = 1.0
        bias = [0.0]
        denominator = [255.0]
    if len(tactile_image.shape) == 3:
        #no batch dimension
        num_channels = tactile_image.shape[0]
        new_tactile_image = torch.zeros_like(tactile_image)
        for i in range(num_channels):
            new_tactile_image[i,...] = (tactile_image[i,...]*denominator[min(i,len(denominator)-1)])/scale + bias[min(i,len(bias)-1)]
    elif len(tactile_image.shape) == 4:
        #batch dimension
        num_channels = tactile_image.shape[1]
        new_tactile_image = torch.zeros_like(tactile_image)
        for i in range(num_channels):
            new_tactile_image[:,i,...] = (tactile_image[:,i,...]*denominator[min(i,len(denominator)-1)])/scale + bias[min(i,len(bias)-1)]
    return new_tactile_image

def normalize_depth_image(depth_image, depth_normalization_method, norm_scale, depth_normalization_params=None):
    if '0_255' not in depth_normalization_method:
        for i in len(depth_normalization_params):
            if i == 0:
                min_depth = depth_normalization_params[i]
            elif i == 1:
                max_depth = depth_normalization_params[i]
            elif i == 2:
                mean_depth = depth_normalization_params[i]
            elif i == 3:
                std_depth = depth_normalization_params[i]
    if depth_normalization_method == 'min_max_to_-1_1':
        scale = norm_scale
        bias = 0.5*(max_depth+min_depth)
        denominator = (max_depth-min_depth)
    elif depth_normalization_method == 'mean_std':
        scale = 1.0
        bias = mean_depth
        denominator = std_depth
    elif depth_normalization_method == 'min_max_to_0_1':
        scale = norm_scale
        bias = min_depth
        denominator = max_depth-min_depth
    elif depth_normalization_method == 'min_max_to_0_-1':
        scale = -norm_scale
        bias = min_depth
        denominator = max_depth-min_depth
    #normalize depth image
    depth_image = scale*(depth_image - bias)/denominator
    return depth_image

def denormalize_depth_image(depth_image, depth_normalization_method, norm_scale, depth_normalization_params=None):
    if '0_255' not in depth_normalization_method:
        for i in len(depth_normalization_params):
            if i == 0:
                min_depth = depth_normalization_params[i]
            elif i == 1:
                max_depth = depth_normalization_params[i]
            elif i == 2:
                mean_depth = depth_normalization_params[i]
            elif i == 3:
                std_depth = depth_normalization_params[i]
    if depth_normalization_method == 'min_max_to_-1_1':
        scale = norm_scale
        bias = 0.5*(max_depth+min_depth)
        denominator = (max_depth-min_depth)
    elif depth_normalization_method == 'mean_std':
        scale = 1.0
        bias = mean_depth
        denominator = std_depth
    elif depth_normalization_method == 'min_max_to_0_1':
        scale = norm_scale
        bias = min_depth
        denominator = max_depth-min_depth
    elif depth_normalization_method == 'min_max_to_0_-1':
        scale = -norm_scale
        bias = min_depth
        denominator = max_depth-min_depth
    #denormalize depth image
    depth_image = (depth_image*denominator)/scale + bias
    return depth_image