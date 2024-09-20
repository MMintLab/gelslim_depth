import torch
import torch.nn.functional as F
from typing import Tuple
import torchvision.transforms.functional as TF

def get_difference_image(tactile_image, base_tactile_image):
    difference_image = tactile_image - base_tactile_image
    #normalize from -255 to 255 to 0 to 255
    difference_image = (difference_image+255.0)/2.0
    return difference_image

def sample_multi_channel_image_to_desired_size(MC_image, desired_size: Tuple[int,int], interp_method='area'):
    #use F.interpolate to sample tactile image or depth image to desired size
    MC_image = F.interpolate(MC_image, size=desired_size, mode=interp_method)
    return MC_image

def blur_depth_images(depth, depth_image_blur_kernel):
    depth = TF.gaussian_blur(depth, kernel_size=depth_image_blur_kernel)
    return depth