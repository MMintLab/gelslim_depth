import torch
import importlib
import os
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

from gelslim_depth.datasets.general_dataset import GeneralDataset

def predict_depth_from_RGB(images, model, output_size):
    min_depth, max_depth = config.depth_normalization_parameters
    images = F.interpolate(images, size=config.input_tactile_image_size, mode='area')/255.0
    depth = model(x=images)
    depth = (max_depth - min_depth)*depth/(-config.norm_scale)+min_depth
    depth = F.interpolate(depth, size=output_size, mode='area')
    return depth

if __name__ == '__main__':
    weights_name = 'unet_new'
    real_data_path = 'data/real_data/'

    pt_file_list = os.listdir(real_data_path)

    gpu = 2

    num_images_from_each_object = 5

    config = importlib.import_module('gelslim_depth.config.config_'+weights_name)

    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')

    if config.model_type == 'unet':
        model = importlib.import_module('gelslim_depth.models.unet').UNet(n_channels=3, n_classes=1, layer_dimensions=config.CNN_dimensions, kernel_size=config.kernel_size, maxpool_size=config.maxpool_size, upconv_stride=config.upconv_stride).to(device)
    
    model.load_state_dict(torch.load(config.weights_path+weights_name+'.pth'))

    model.eval()

    num_objects = len(pt_file_list)

    fig, axs = plt.subplots(num_images_from_each_object, num_objects*2, figsize=(2*num_objects, 1.1*num_images_from_each_object))

    for i in range(num_objects):
        right_or_left = torch.randint(0, 2, (num_images_from_each_object,))
        pt_file = pt_file_list[i]
        pt = torch.load(real_data_path+pt_file, map_location='cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
        pt_length = pt['tactile_image'].size()[0]
        indices = torch.randint(0, pt_length, (num_images_from_each_object,))
        
        tactile_images = torch.zeros((num_images_from_each_object, 3, pt['tactile_image'].size()[2],  pt['tactile_image'].size()[3])).to(device)
        for j in range(num_images_from_each_object):
            if config.use_difference_image:
                tactile_images[j, ...] = (pt['tactile_image'][indices[j], right_or_left[j]*3:right_or_left[j]*3+3, ...]-pt['base_tactile_image'][indices[j], right_or_left[j]*3:right_or_left[j]*3+3, ...])/2.0 + 127.5
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
    
    #tight layout
    plt.tight_layout()
    fig.savefig('test_output/real_data_depth_predictions.png')