import sys
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

data_name = sys.argv[1]

data_dir = '/data/william/gelslim_depth/data/real_data'

#get list of files in the data directory
files = os.listdir(data_dir)

#select filenames that contain the data_name
data_files = [file for file in files if data_name in file]
data_file = data_dir+'/'+data_files[0]

pt = torch.load(data_file)

tactile_images = pt['tactile_image']
in_hand_poses = pt['in_hand_pose']
try:
    depth_images = pt['depth_image']
except:
    depth_images = torch.zeros(tactile_images.shape[0], 2, tactile_images.shape[2], tactile_images.shape[3])

num_images = tactile_images.shape[0]

print('Found', num_images, 'data points')

min_depth = 0

#in a while true loop, randomly select 5 points from the dataset, plot tactile images alongside the in_hand_pose and depth images for each
while True:
    indices = np.random.choice(num_images, 5, replace=False)

    fig, axs = plt.subplots(5, 4, figsize=(20,25))
    for i in range(5):
        left_tactile_image = tactile_images[indices[i],:3,:,:].permute(1,2,0).numpy().astype(np.uint8)
        right_tactile_image = tactile_images[indices[i],3:,:,:].permute(1,2,0).numpy().astype(np.uint8)
        left_depth_image = depth_images[indices[i],:1,:,:].permute(1,2,0).numpy()
        right_depth_image = depth_images[indices[i],1:,:,:].permute(1,2,0).numpy()
        min_depth = np.min([min_depth, np.min(left_depth_image), np.min(right_depth_image)])
        in_hand_pose = in_hand_poses[indices[i],:].numpy()
        axs[i, 0].imshow(left_tactile_image)
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        LD = axs[i, 1].imshow(left_depth_image)
        #fig.colorbar(im=LD, ax=axs[i, 1])
        axs[i, 1].set_xticks([])
        axs[i, 1].set_yticks([])
        axs[i, 1].set_title("In Hand Pose: "+str((1000*in_hand_pose[0]).round(1))+" mm, "+str((1000*in_hand_pose[1]).round(1))+" mm, "+str((180*(1/np.pi)*in_hand_pose[2]).round(1))+" deg")

        axs[i, 2].imshow(right_tactile_image, vmax=0, vmin=min_depth)
        axs[i, 2].set_xticks([])
        axs[i, 2].set_yticks([])
        RD = axs[i, 3].imshow(right_depth_image, vmax=0, vmin=min_depth)
        #fig.colorbar(im=RD, ax=axs[i, 3])
        axs[i, 3].set_xticks([])
        axs[i, 3].set_yticks([])

        fig.suptitle("Left                    Right")

    #save the figure
    plt.savefig('scripts/data_scripts/pt_images/'+data_name+'.png', dpi=300)
    input('Press Enter to continue')