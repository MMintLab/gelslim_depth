import torch
import importlib
import os
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

from gelslim_depth.processing_utils.normalization_utils import denormalize_depth_image, normalize_tactile_image
from gelslim_depth.processing_utils.image_utils import sample_multi_channel_image_to_desired_size, get_difference_image
import gelslim_depth.main_config as main_config
import sys

def predict_depth_from_RGB(images, model, output_size):
    images = sample_multi_channel_image_to_desired_size(images, config.input_tactile_image_size, config.interp_method)
    images = normalize_tactile_image(images, config.tactile_normalization_method, config.norm_scale, config.tactile_normalization_parameters)
    depth = model(x=images)
    depth = denormalize_depth_image(depth, config.depth_normalization_method, config.norm_scale, config.depth_normalization_parameters)
    depth = sample_multi_channel_image_to_desired_size(depth, output_size, config.interp_method)
    return depth

if __name__ == '__main__':
    weights_name = sys.argv[1]

    gpu = sys.argv[2]

    sub_dir = sys.argv[3]

    data_path = main_config.DATA_PATH+'/'+sub_dir+'/'

    #gpu to use

    pt_file_list = os.listdir(data_path)

    #remove non .pt files
    pt_file_list = [pt_file for pt_file in pt_file_list if pt_file[-3:] == '.pt']

    #arguments are the list of pt files to display
    if len(sys.argv) > 4:
        object_list = sys.argv[4:]

        new_pt_file_list = []
        for object_name in object_list:
            for pt_file in pt_file_list:
                if object_name in pt_file:
                    new_pt_file_list.append(pt_file)
        pt_file_list = new_pt_file_list

    #max_number of objects to display
    max_num_objects = 5
    pt_file_list = pt_file_list[:max_num_objects]

    #number of images to display from each object
    num_images_from_each_object = 5

    config = importlib.import_module('gelslim_depth.config.config_'+weights_name)

    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')

    if config.model_type == 'unet':
        model = importlib.import_module('gelslim_depth.models.unet').UNet(n_channels=3, n_classes=1, layer_dimensions=config.CNN_dimensions, kernel_size=config.kernel_size, maxpool_size=config.maxpool_size, upconv_stride=config.upconv_stride).to(device)
    
    model.load_state_dict(torch.load(config.weights_path+weights_name+'.pth', map_location=device))

    model.eval()

    num_objects = len(pt_file_list)

    fig, axs = plt.subplots(num_images_from_each_object, num_objects*2, figsize=(2*num_objects, 1.1*num_images_from_each_object))

    for i in range(num_objects):
        right_or_left = torch.randint(0, 2, (num_images_from_each_object,))
        pt_file = pt_file_list[i]
        print("Testing on: ", pt_file)
        pt = torch.load(data_path+pt_file, map_location='cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
        pt_length = pt['tactile_image'].size()[0]
        indices = torch.randint(0, pt_length, (num_images_from_each_object,))
        
        tactile_images = torch.zeros((num_images_from_each_object, 3, pt['tactile_image'].size()[2],  pt['tactile_image'].size()[3])).to(device)
        num_base_tactile_images = pt['base_tactile_image'].size()[0]
        for j in range(num_images_from_each_object):
            if config.use_difference_image:
                tactile_images[j, ...] = get_difference_image(pt['tactile_image'][indices[j], right_or_left[j]*3:right_or_left[j]*3+3, ...],pt['base_tactile_image'][min(indices[j],num_base_tactile_images), right_or_left[j]*3:right_or_left[j]*3+3, ...])
            else:
                tactile_images[j, ...] = pt['tactile_image'][indices[j], right_or_left[j]*3:right_or_left[j]*3+3, ...]

        depth_images = predict_depth_from_RGB(tactile_images, model, output_size=(pt['tactile_image'].size()[2], pt['tactile_image'].size()[3]))

        for j in range(num_images_from_each_object):
            tactile_image = tactile_images[j, ...].permute(1,2,0).cpu().numpy().astype(np.uint8)
            depth_image = depth_images[j, ...].permute(1,2,0).detach().cpu().numpy()
            axs[j, 2*i].imshow(tactile_image)
            axs[j, 2*i].set_xticks([])
            axs[j, 2*i].set_yticks([])
            axs[j, 2*i+1].imshow(depth_image)
            axs[j, 2*i+1].set_xticks([])
            axs[j, 2*i+1].set_yticks([])
    
    #make directory to save images
    if not os.path.exists('test_output'):
        os.makedirs('test_output')
    #tight layout
    plt.tight_layout()
    fig.savefig('test_output/depth_predictions.png')