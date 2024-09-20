import torch

import sys
import rospy
from gelslim_utils.camera_parsers.gelslim_camera_parser import GelslimCameraParser
from gelslim_depth.processing_utils.image_utils import sample_multi_channel_image_to_desired_size, get_difference_image
from gelslim_depth.processing_utils.normalization_utils import normalize_tactile_image, denormalize_depth_image

from wsg_50_utils.wsg_50_gripper import WSG50Gripper

import numpy as np

from sensor_msgs.msg import Image

import importlib

import cv2

def force_gripper_to_width(gripper, width):
    gripper.move(width=width)
    rospy.sleep(1.0)
    while gripper.get_width() < width-5 or gripper.get_width() > width+5:
        print('Gripper not at correct width, trying to open again')
        gripper.move(width=width)
        rospy.sleep(2.0)

if __name__ == '__main__':
    rospy.init_node('depth_broadcaster', anonymous=True)

    weights_name = sys.argv[1]
    config = importlib.import_module('gelslim_depth.config.config_'+weights_name)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config.model_type == 'unet':
        model = importlib.import_module('gelslim_depth.models.unet').UNet(n_channels=3, n_classes=1, layer_dimensions=config.CNN_dimensions, kernel_size=config.kernel_size, maxpool_size=config.maxpool_size, upconv_stride=config.upconv_stride).to(device)
    
    model.load_state_dict(torch.load(config.weights_path+weights_name+'.pth', map_location=device))

    model.eval()

    left_gelslim = GelslimCameraParser(camera_name='gelslim_left', verbose=False)
    print('Left gelslim connected')
    right_gelslim = GelslimCameraParser(camera_name='gelslim_right', verbose=False)
    print('Right gelslim connected')

    left_pub = rospy.Publisher('left_gelslim_depth', Image, queue_size=1)
    right_pub = rospy.Publisher('right_gelslim_depth', Image, queue_size=1)

    left_image_msg = Image()
    right_image_msg = Image()

    if config.use_difference_image:
        print('need base image, opening gripper to collect base image')
        gripper = WSG50Gripper()
        #open gripper
        force_gripper_to_width(gripper, 100)
        rospy.sleep(1.5)
        #collect base image
        left_gelslim_image = left_gelslim.get_image_color()
        right_gelslim_image = right_gelslim.get_image_color()
        #concatenate images
        base_tactile_image = torch.from_numpy(np.concatenate((left_gelslim_image, right_gelslim_image), axis=2).astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
        print('Base image collected')

    while not rospy.is_shutdown():
        left_image = left_gelslim.get_image_color()
        right_image = right_gelslim.get_image_color()
        tactile_image = torch.from_numpy(np.concatenate((left_image, right_image), axis=2).astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
        if config.use_difference_image:
            tactile_image = get_difference_image(tactile_image, base_tactile_image)
        tactile_image = sample_multi_channel_image_to_desired_size(tactile_image, config.input_tactile_image_size)
        print("min: ", tactile_image.min())
        print("max: ", tactile_image.max())
        tactile_image = normalize_tactile_image(tactile_image, '0_255_to_0_1', config.norm_scale)
        left_vis_final = cv2.cvtColor(((tactile_image[:,:3,...].squeeze(0).permute(1,2,0).detach().cpu().numpy())).astype(np.float32), cv2.COLOR_RGB2BGR)
        right_vis_final = cv2.cvtColor(((tactile_image[:,3:,...].squeeze(0).permute(1,2,0).detach().cpu().numpy())).astype(np.float32), cv2.COLOR_RGB2BGR)
        side_by_side_final = np.concatenate((left_vis_final, right_vis_final), axis=1)
        side_by_side_final = cv2.resize(side_by_side_final, dsize=(int(config.input_tactile_image_size[1]*10*2), int(config.input_tactile_image_size[0]*10)), interpolation=cv2.INTER_NEAREST)

        #cv2.imshow('tactile_image', side_by_side_final)
        #cv2.waitKey(1)
        #import pdb; pdb.set_trace()
        #denorm_tactile_image = tactile_image*255.0
        #left_vis_final = cv2.cvtColor(((denorm_tactile_image[:,:3,...].squeeze(0).permute(1,2,0).detach().cpu().numpy())).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #right_vis_final = cv2.cvtColor(((denorm_tactile_image[:,3:,...].squeeze(0).permute(1,2,0).detach().cpu().numpy())).astype(np.uint8), cv2.COLOR_RGB2BGR)
        #side_by_side_final = np.concatenate((left_vis_final, right_vis_final), axis=1)
        #blow up the image 5x
        #side_by_side_final = cv2.resize(side_by_side_final, dsize=(int(config.input_tactile_image_size[1]*10*2), int(config.input_tactile_image_size[0]*10)), interpolation=cv2.INTER_NEAREST)
        #cv2.imshow('tactile_image', side_by_side_final)
        #cv2.waitKey(1)
        left_depth = model(tactile_image[:, :3, ...])
        right_depth = model(tactile_image[:, 3:, ...])
        #left_depth = denormalize_depth_image(left_depth, 'min_max_to_0_-1', config.norm_scale, config.depth_normalization_parameters)
        #right_depth = denormalize_depth_image(right_depth, 'min_max_to_0_-1', config.norm_scale, config.depth_normalization_parameters)
        left_depth = sample_multi_channel_image_to_desired_size(left_depth, (left_image.shape[0], left_image.shape[1]))
        right_depth = sample_multi_channel_image_to_desired_size(right_depth, (right_image.shape[0], right_image.shape[1]))
        left_depth = (left_depth.squeeze(0).permute(1,2,0).detach().cpu().numpy()*(-255.0)).astype(np.uint8)
        right_depth = (right_depth.squeeze(0).permute(1,2,0).detach().cpu().numpy()*(-255.0)).astype(np.uint8)
        left_image_msg.data = left_depth.tobytes()
        right_image_msg.data = right_depth.tobytes()
        left_image_msg.header.stamp = rospy.Time.now()
        right_image_msg.header.stamp = rospy.Time.now()
        left_image_msg.height = left_depth.shape[0]
        left_image_msg.width = left_depth.shape[1]
        left_image_msg.encoding = 'mono8'
        left_image_msg.step = left_depth.shape[1]
        right_image_msg.height = right_depth.shape[0]
        right_image_msg.width = right_depth.shape[1]
        right_image_msg.encoding = 'mono8'
        right_image_msg.step = right_depth.shape[1]
        left_pub.publish(left_image_msg)
        right_pub.publish(right_image_msg)