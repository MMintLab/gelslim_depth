import torch
import importlib
import os
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

from tqdm import tqdm

from gelslim_depth.datasets.general_dataset import GeneralDataset

def predict_depth_from_RGB(images, model, output_size):
    min_depth, max_depth = config.depth_normalization_parameters
    images = F.interpolate(images, size=config.input_tactile_image_size, mode='area')/255.0
    depth = model(x=images)
    depth = (max_depth - min_depth)*depth/(-config.norm_scale)+min_depth
    depth = F.interpolate(depth, size=output_size, mode='area')
    return depth

if __name__ == '__main__':
    max_num_images = 50
    weights_name = 'unet_bigdata'
    real_data_path = '/data/william/gelslim_depth/data/real_data/'
    train_data_path = '/data/william/gelslim_depth/data/train_data/'
    test_data_path = '/data/william/gelslim_depth/data/test_data/'
    validation_data_path = '/data/william/gelslim_depth/data/validation_data/'

    #get list of validation objects from validation_objects.txt
    with open('/data/william/gelslim_depth/data/validation_objects.txt', 'r') as f:
        validation_objects = f.readlines()
    validation_objects = [object.strip() for object in validation_objects]

    #get list of test objects from test_objects.txt
    with open('/data/william/gelslim_depth/data/test_objects.txt', 'r') as f:
        test_objects = f.readlines()
    test_objects = [object.strip() for object in test_objects]

    #get list of real train objects from train_real_objects.txt
    with open('/data/william/gelslim_depth/data/real_data/train_real_objects.txt', 'r') as f:
        train_real_objects = f.readlines()
    train_real_objects = [object.strip() for object in train_real_objects]

    seen_grasps_paths = []
    unseen_grasps_paths = []
    unseen_objects_paths = []

    real_data_path_pt_list = os.listdir(real_data_path)
    train_data_path_pt_list = os.listdir(train_data_path)
    test_data_path_pt_list = os.listdir(test_data_path)
    validation_data_path_pt_list = os.listdir(validation_data_path)

    #remove non .pt files
    real_data_path_pt_list = [pt_file for pt_file in real_data_path_pt_list if pt_file[-3:] == '.pt']
    train_data_path_pt_list = [pt_file for pt_file in train_data_path_pt_list if pt_file[-3:] == '.pt']
    test_data_path_pt_list = [pt_file for pt_file in test_data_path_pt_list if pt_file[-3:] == '.pt']
    #remove all test files that do not contain 'peg1'
    test_data_path_pt_list = [pt_file for pt_file in test_data_path_pt_list if 'peg1' in pt_file]
    validation_data_path_pt_list = [pt_file for pt_file in validation_data_path_pt_list if pt_file[-3:] == '.pt']

    for pt_file in real_data_path_pt_list:
        #if the real data object is in real train objects then add to seen_grasps_paths
        if pt_file.split('.')[0] in train_real_objects:
            seen_grasps_paths.append(real_data_path+pt_file)
        else:
            unseen_objects_paths.append(real_data_path+pt_file)

    for pt_file in train_data_path_pt_list:
        #if the object is in validation objects or test objects then add to unseen_grasps_paths
        if pt_file.split('_train')[0] in validation_objects or pt_file.split('.')[0] in test_objects:
            unseen_grasps_paths.append(train_data_path+pt_file)
        else:
            seen_grasps_paths.append(train_data_path+pt_file)

    for pt_file in test_data_path_pt_list:
        #if the object is in test objects then add to unseen_objects_paths
        if pt_file.split('_test')[0] in test_objects or pt_file.split('_test')[0] in validation_objects:
            if pt_file.split('_test')[0] in test_objects:
                unseen_objects_paths.append(test_data_path+pt_file)
        else:
            unseen_grasps_paths.append(test_data_path+pt_file)

    gpu = 2

    config = importlib.import_module('gelslim_depth.config.config_'+weights_name)

    min_depth, max_depth = config.depth_normalization_parameters

    device = torch.device('cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')

    if config.model_type == 'unet':
        model = importlib.import_module('gelslim_depth.models.unet').UNet(n_channels=3, n_classes=1, layer_dimensions=config.CNN_dimensions, kernel_size=config.kernel_size, maxpool_size=config.maxpool_size, upconv_stride=config.upconv_stride).to(device)
    
    model.load_state_dict(torch.load(config.weights_path+weights_name+'.pth'))

    model.eval()

    folders_to_avoid = ['/train_data/']

    fig, axs = plt.subplots(1, 2, figsize=(2, 1.1))

    seen_grasps_paths = []

    print('Predicting depth for seen grasps')
    for i in range(len(seen_grasps_paths)):
        avoid = False
        for folder in folders_to_avoid:
            if folder in seen_grasps_paths[i]:
                avoid = True
                break
        if avoid:
            continue
        object_name = seen_grasps_paths[i].split('/')[-1].replace('_train.pt','').replace('_test.pt','').replace('.pt','')
        right_or_left = 0
        print("Loading seen grasp path: ", seen_grasps_paths[i])
        pt = torch.load(seen_grasps_paths[i], map_location='cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
        print("Loaded seen grasp path: ", seen_grasps_paths[i])
        pt_length = pt['tactile_image'].size()[0]
        if pt_length > max_num_images:
            indices = torch.randint(0, pt_length, (max_num_images,))
        else:
            indices = torch.arange(pt_length)
        
        tactile_images = torch.zeros((max_num_images, 3, pt['tactile_image'].size()[2],  pt['tactile_image'].size()[3])).to(device)
        if config.use_difference_image:
            tactile_images = (pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]-pt['base_tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...])/2.0 + 127.5
        else:
            tactile_images = pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]
        depth_images = predict_depth_from_RGB(tactile_images, model, output_size=(pt['tactile_image'].size()[2], pt['tactile_image'].size()[3]))

        for j in tqdm(range(min(pt_length, max_num_images))):
            tactile_image = tactile_images[j, ...].permute(1,2,0).cpu().numpy().astype(np.uint8)
            depth_image = depth_images[j, ...].permute(1,2,0).detach().cpu().numpy()
            axs[0].imshow(tactile_image)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            #use cmap binary
            axs[1].imshow(depth_image, cmap='binary', vmin=min_depth, vmax=max_depth)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
    
            #tight layout
            plt.tight_layout()
            fig.savefig('/data/william/gelslim_depth/test_output/seen_grasps/'+object_name+'_'+str(j)+'.png', dpi=300)
            axs[0].cla()
            axs[1].cla()

    print('Predicting depth for unseen grasps')
    print(unseen_grasps_paths)
    import pdb; pdb.set_trace()

    for i in range(len(unseen_grasps_paths)):
        avoid = False
        for folder in folders_to_avoid:
            if folder in unseen_grasps_paths[i]:
                avoid = True
                break
        if avoid:
            continue
        object_name = unseen_grasps_paths[i].split('/')[-1].replace('_train.pt','').replace('_test.pt','').replace('.pt','')
        right_or_left = 0
        print("Loading unseen grasp path: ", unseen_grasps_paths[i])
        pt = torch.load(unseen_grasps_paths[i], map_location='cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
        print("Loaded unseen grasp path: ", unseen_grasps_paths[i])
        pt_length = pt['tactile_image'].size()[0]
        if pt_length > max_num_images:
            indices = torch.randint(0, pt_length, (max_num_images,))
        else:
            indices = torch.arange(0, pt_length)
        
        tactile_images = torch.zeros((max_num_images, 3, pt['tactile_image'].size()[2],  pt['tactile_image'].size()[3])).to(device)
        if config.use_difference_image:
            tactile_images = (pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]-pt['base_tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...])/2.0 + 127.5
        else:
            tactile_images = pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]
        depth_images = predict_depth_from_RGB(tactile_images, model, output_size=(pt['tactile_image'].size()[2], pt['tactile_image'].size()[3]))

        for j in tqdm(range(min(pt_length, max_num_images))):
            tactile_image = tactile_images[j, ...].permute(1,2,0).cpu().numpy().astype(np.uint8)
            depth_image = depth_images[j, ...].permute(1,2,0).detach().cpu().numpy()
            axs[0].imshow(tactile_image)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            #use cmap binary
            axs[1].imshow(depth_image, cmap='binary', vmin=min_depth, vmax=max_depth)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
    
            #tight layout
            plt.tight_layout()
            fig.savefig('/data/william/gelslim_depth/test_output/unseen_grasps/'+object_name+'_'+str(j)+'.png', dpi=300)
            axs[0].cla()
            axs[1].cla()

    print('Predicting depth for unseen objects')

    for i in range(len(unseen_objects_paths)):
        avoid = False
        for folder in folders_to_avoid:
            if folder in unseen_objects_paths[i]:
                avoid = True
                break
        if avoid:
            continue
        object_name = unseen_objects_paths[i].split('/')[-1].replace('_train.pt','').replace('_test.pt','').replace('.pt','')
        right_or_left = 0
        pt = torch.load(unseen_objects_paths[i], map_location='cuda:'+str(gpu) if torch.cuda.is_available() else 'cpu')
        pt_length = pt['tactile_image'].size()[0]
        if pt_length > max_num_images:
            indices = torch.randint(0, pt_length, (max_num_images,))
        else:
            indices = torch.arange(0, pt_length)
        
        tactile_images = torch.zeros((max_num_images, 3, pt['tactile_image'].size()[2],  pt['tactile_image'].size()[3])).to(device)

        if config.use_difference_image:
            tactile_images = (pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]-pt['base_tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...])/2.0 + 127.5
        else:
            tactile_images = pt['tactile_image'][indices, right_or_left*3:right_or_left*3+3, ...]

        depth_images = predict_depth_from_RGB(tactile_images, model, output_size=(pt['tactile_image'].size()[2], pt['tactile_image'].size()[3]))

        for j in tqdm(range(min(pt_length, max_num_images))):
            tactile_image = tactile_images[j, ...].permute(1,2,0).cpu().numpy().astype(np.uint8)
            depth_image = depth_images[j, ...].permute(1,2,0).detach().cpu().numpy()
            axs[0].imshow(tactile_image)
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            #use cmap binary
            axs[1].imshow(depth_image, cmap='binary', vmin=min_depth, vmax=max_depth)
            axs[1].set_xticks([])
            axs[1].set_yticks([])
    
            #tight layout
            plt.tight_layout()
            fig.savefig('/data/william/gelslim_depth/test_output/unseen_objects/'+object_name+'_'+str(j)+'.png', dpi=300)
            axs[0].cla()
            axs[1].cla()

    print('Finished predicting depth for seen grasps, unseen grasps, and unseen objects')
    print('Images saved to /data/william/gelslim_depth/test_output/')
