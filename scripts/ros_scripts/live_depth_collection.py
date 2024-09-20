import torch
from gelslim_depth.models.unet import UNet

import numpy as np

import os
import sys
import rospy
from bubble_utils.bubble_med.bubble_med import BubbleMed
from gelslim_utils.grasp_utils.wsg50_grasp_control import GelSlimGraspControl

from gelslim_utils.camera_parsers.gelslim_camera_parser import GelslimCameraParser

from tactile_diffusion.dataset_utils.force_recorder import WrenchListener
from wsg_50_utils.wsg_50_gripper import WSG50Gripper

from std_msgs.msg import Bool

import tf
import tf.transformations as tr
from tqdm import tqdm

data_name = sys.argv[1]

num_images_to_collect = int(sys.argv[2])

start_index = 0

period = 0.5

open_time = 1.0

max_num_images_to_collect_while_closed = 10

min_num_images_to_collect_while_closed = 1

holding_joint_pose = np.array([0.09422932395076782, 0.5145552797570093, -0.18115042777941665, -1.5700313137499131, 1.3516604601272098, 0.1725619929429966, 0.349385576173708])

def force_gripper_to_width(gripper, width):
    gripper.move(width=width)
    rospy.sleep(1.0)
    while gripper.get_width() < width-5 or gripper.get_width() > width+5:
        print('Gripper not at correct width, trying to open again')
        gripper.move(width=width)
        rospy.sleep(2.0)

if __name__ == '__main__':
    rospy.init_node('in_hand_pose_collection', anonymous=True)

    left_gelslim = GelslimCameraParser(camera_name='gelslim_left', verbose=False)
    print('Left gelslim connected')
    right_gelslim = GelslimCameraParser(camera_name='gelslim_right', verbose=False)
    print('Right gelslim connected')

    tf_listener = tf.TransformListener()

    data_directory = '/home/william/robot_ws/src/in_hand_pose/ood_data/'+data_name+'/'
    print(data_directory)
    #if data directory does not exist, create it
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
        os.makedirs(data_directory+'in_hand_pose/')
        os.makedirs(data_directory+'data_view/')
        os.makedirs(data_directory+'raw_gelslim/')
        os.makedirs(data_directory+'max_divergence/')
        os.makedirs(data_directory+'gripper_position/')
        os.makedirs(data_directory+'force/')
    
    robot = BubbleMed()
    robot.connect()
    gripper = WSG50Gripper()

    grasp_control = GelSlimGraspControl(minimum_distance=15, divergence_threshold=0.6, gripper_step_size=0.2, speed=50.0)

    max_divergence_list = []
    gripper_position_list = []

    tf_listener = tf.TransformListener()

    force_sub = WrenchListener('/med/wrench')
    print('Force sensor connected')

    #open gripper
    force_gripper_to_width(gripper, 35)

    rospy.sleep(1.5)
    left_gelslim_image = left_gelslim.get_image_color()
    right_gelslim_image = right_gelslim.get_image_color()
    #concatenate images
    full_image = np.concatenate((left_gelslim_image, right_gelslim_image), axis=2)
    #save images as npy files
    np.save(data_directory+'base_raw_gelslim.npy', full_image)

    #lift arm
    cp = robot.get_current_pose()
    cp[2] += 0.2
    robot.set_pose(cp)
    rospy.sleep(1.0)

    #move to holding joint pose
    robot.plan_to_joint_config(group_name=robot.arm_group, joint_config=list(holding_joint_pose))

    input('Press enter to start collecting data')

    num_images_collected = 0+start_index

    while num_images_collected <= num_images_to_collect:
        num_images_to_collect_while_closed = np.random.randint(min_num_images_to_collect_while_closed, max_num_images_to_collect_while_closed)
        gripper_position, max_divergence, _, _ = grasp_control.grasp_until_divergence()
        rospy.sleep(0.5)
        for i in tqdm(range(num_images_to_collect_while_closed)):
            in_hand_pose = np.array([0.0, 0.0, 0.0])
            tf_listener.waitForTransform('grasp_frame', 'apriltag_object_frame', rospy.Time(), rospy.Duration(4.0))
            object_pose_wrt_grasp = tf_listener.lookupTransform('grasp_frame', 'apriltag_object_frame', rospy.Time(0))
            print('Object pose wrt grasp: '+str(object_pose_wrt_grasp))
            y_pos = object_pose_wrt_grasp[0][1]
            z_pos = object_pose_wrt_grasp[0][2]
            ax, ay, az = tr.euler_from_quaternion(object_pose_wrt_grasp[1])
            in_hand_pose[0] = y_pos
            in_hand_pose[1] = z_pos
            in_hand_pose[2] = ax

            print('thetas: '+str(in_hand_pose[2]*180/np.pi)+' degrees')

            wrench = force_sub.get_wrench()
            left_gelslim_image = left_gelslim.get_image_color()
            right_gelslim_image = right_gelslim.get_image_color()
            #concatenate images
            full_image = np.concatenate((left_gelslim_image, right_gelslim_image), axis=2)
            #save images as npy files
            np.save(data_directory+'raw_gelslim/'+str(num_images_collected)+'.npy', full_image)
            #save in hand pose
            np.save(data_directory+'in_hand_pose/'+str(num_images_collected)+'.npy', in_hand_pose)
            #save force sensor data
            np.save(data_directory+'force/'+str(num_images_collected)+'.npy', wrench)
            #save max divergence
            np.save(data_directory+'max_divergence/'+str(num_images_collected)+'.npy', max_divergence)
            max_divergence_list.append(max_divergence)
            #save gripper position
            np.save(data_directory+'gripper_position/'+str(num_images_collected)+'.npy', gripper_position)
            gripper_position_list.append(gripper_position)
            num_images_collected += 1
            rospy.sleep(period)
            print('Saved image for pose: y='+str(in_hand_pose[0])+', z='+str(in_hand_pose[1])+', th='+str(in_hand_pose[2]))
        rospy.sleep(0.2)
        force_gripper_to_width(gripper, 35)
        rospy.sleep(open_time)

    #save max divergence and gripper position lists as txt files separated by newlines
    np.savetxt(data_directory+'max_divergence.txt', max_divergence_list, delimiter='\n')
    np.savetxt(data_directory+'gripper_position.txt', gripper_position_list, delimiter='\n')
    print('Finished collecting data')
    #print stats on max divergence and gripper position
    print('Max divergence max: '+str(np.max(max_divergence_list)))
    print('Max divergence mean: '+str(np.mean(max_divergence_list)))
    print('Max divergence min: '+str(np.min(max_divergence_list)))
    print('Gripper position max: '+str(np.max(gripper_position_list)))
    print('Gripper position mean: '+str(np.mean(gripper_position_list)))
    print('Gripper position min: '+str(np.min(gripper_position_list)))