import torch
import open3d as o3d
import numpy as np
import scipy.interpolate as interp
import os
from tqdm import tqdm

class DepthImageGenerator():
    def __init__(self, mesh_dir, object_list, pc_scale, dataset_dir, grasp_widths_file, gelslim_plane='+y+z', LR_flip=False, image_size=(320,427), image_height_mm=12, grasp_width_offset=0.0, pc_sampling=1e5, device='cpu'):
        self.image_height_mm = image_height_mm
        self.image_size = image_size
        self.mm_per_pixel = image_height_mm / image_size[0]
        self.mesh_dir = mesh_dir
        self.grasp_widths_file = grasp_widths_file
        self.gelslim_plane = gelslim_plane #(for right image)
        self.LR_flip = LR_flip
        self.pc_scale = pc_scale
        self.dataset_dir = dataset_dir
        self.object_list = object_list
        self.plane_axes = [c for c in self.gelslim_plane if c.isalpha()]
        self.pc_sampling = pc_sampling
        self.device = torch.device(device)
        self.grasp_width_offset = grasp_width_offset

    def generate_depth_images_v1(self):
        dataset_list = os.listdir(self.dataset_dir)
        dataset_list = [f for f in dataset_list if f[-3:] == '.pt']
        if self.object_list is not None:
            #remove files that are not in the object list
            if '_val' in dataset_list[0] or '_test' in dataset_list[0] or '_train' in dataset_list[0]:
                dataset_list = [f for f in dataset_list if f.split('_')[-2] in self.object_list]
            else:
                dataset_list = [f for f in dataset_list if f.split('.')[0] in self.object_list]
        user_in = input('Generating depth images for ' + str(dataset_list) + ', Press enter to continue or q to quit.')
        if user_in == 'q':
            return
        #load the grasp widths from the txt file with the format "object: inter_gelslim_distance \n"and convert to a dictionary 
        with open(self.grasp_widths_file, 'r') as f:
            grasp_width_file = f.readlines()
        grasp_width_file = [line.split(':') for line in grasp_width_file]
        grasp_widths = {}
        for line in grasp_width_file:
            if line[1] != ' None\n' and line[1] != ' None':
                grasp_widths[line[0]] = float(line[1])
            else:
                grasp_widths[line[0]] = None
        for pt_file in dataset_list:
            dataset_pt = torch.load(self.dataset_dir + '/' + pt_file, map_location=self.device)
            num_datapoints = dataset_pt['tactile_image'].shape[0]
            #mesh  filename replaces .pt with .stl and removes '_val' or '_test' or '_train'
            if '_val' in pt_file or '_test' in pt_file or '_train' in pt_file:
                mesh_filename = pt_file.split('_')[-2] + '.stl'
            else:
                mesh_filename = pt_file.split('.')[0] + '.stl'
            mesh = o3d.io.read_triangle_mesh(self.mesh_dir + '/' + mesh_filename)
            pc = mesh.sample_points_uniformly(int(self.pc_sampling))
            pc = torch.from_numpy(np.array(pc.points).astype(np.float32)).to(self.device)
            pc = pc * self.pc_scale
            dataset_pt['depth_image'] = torch.zeros((num_datapoints, 2, self.image_size[0], self.image_size[1]))
            for i in tqdm(range(num_datapoints)):
                in_hand_pose = dataset_pt['in_hand_pose'][i, ...]
                if '_val' in pt_file or '_test' in pt_file or '_train' in pt_file:
                    key = pt_file.split('_')[-2]
                else:
                    key = pt_file.split('.')[0]
                inter_gelslim_distance = grasp_widths[key]
                if inter_gelslim_distance is None:
                    inter_gelslim_distance = dataset_pt['grasp_widths'][i]
                inter_gelslim_distance += self.grasp_width_offset
                right_depth_image, left_depth_image = self.generate_depth_image(pc, in_hand_pose[0], in_hand_pose[1], in_hand_pose[2], inter_gelslim_distance, invert_affine=False)
                right_depth_image = right_depth_image.unsqueeze(0)
                left_depth_image = left_depth_image.unsqueeze(0)
                if self.LR_flip:
                    depth_image = torch.cat((right_depth_image, left_depth_image), axis=0)
                else:
                    depth_image = torch.cat((left_depth_image, right_depth_image), axis=0)
                dataset_pt['depth_image'][i, ...] = depth_image
            torch.save(dataset_pt, self.dataset_dir + '/' + pt_file)

    def generate_depth_image(self, pc, translation1, translation2, angle, inter_gelslim_distance, invert_affine=False):
        #use invert_affine=True if translation1, translation2, and angle are the values of the grasp frame with respect to the point cloud frame
        #use invert_affine=False if translation1, translation2, and angle are the values of the point cloud frame with respect to the grasp frame (i.e. "in_hand_pose"), this is the default and compatible with the provided dataset
        #self.plane_axes removes + and - from the gelslim_plane string and returns a list of the characters in the string
        #plane_signs removes the characters in the gelslim_plane string that are not + or - and returns a list of the characters in the string
        plane_signs = [c for c in self.gelslim_plane if c in ['+', '-']]
        # Truncate the point cloud to the gelslim plane, 1/2 of the gelslim distance from 0 in the perpendicular direction
        if 'x' in self.plane_axes and 'y' in self.plane_axes:
            perp_ind = 2
            if self.plane_axes[0] == 'x' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '+z'
                aligned_index = 1
                unaligned_index = 0
            elif self.plane_axes[0] == 'x' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '-z'
                aligned_index = 1
                unaligned_index = 0
            elif self.plane_axes[0] == 'y' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '-z'
                aligned_index = 0
                unaligned_index = 1
            elif self.plane_axes[0] == 'y' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '+z'
                aligned_index = 0
                unaligned_index = 1
        elif 'x' in self.plane_axes and 'z' in self.plane_axes:
            perp_ind = 1
            if self.plane_axes[0] == 'x' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '-y'
                aligned_index = 2
                unaligned_index = 0
            elif self.plane_axes[0] == 'x' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '+y'
                aligned_index = 2
                unaligned_index = 0
            elif self.plane_axes[0] == 'z' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '+y'
                aligned_index = 0
                unaligned_index = 2
            elif self.plane_axes[0] == 'z' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '-y'
                aligned_index = 0
                unaligned_index = 2
        elif 'y' in self.plane_axes and 'z' in self.plane_axes:
            perp_ind = 0
            if self.plane_axes[0] == 'y' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '+x'
                aligned_index = 2
                unaligned_index = 1
            elif self.plane_axes[0] == 'y' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '-x'
                aligned_index = 2
                unaligned_index = 1
            elif self.plane_axes[0] == 'z' and plane_signs[0] == plane_signs[1]:
                right_out_of_plane_dir = '-x'
                aligned_index = 1
                unaligned_index = 2
            elif self.plane_axes[0] == 'z' and plane_signs[0] != plane_signs[1]:
                right_out_of_plane_dir = '+x'
                aligned_index = 1
                unaligned_index = 2
        else:
            raise ValueError('Invalid gelslim_plane')
        if '+' in  right_out_of_plane_dir:
            multiplier = 1
        elif '-' in right_out_of_plane_dir:
            multiplier = -1
        #convention here is angle is about the positive axis in the out of plane direction
        #and that the axis of translation1 and the axis of translation2 follow the right hand rule (their cross product is the out of plane direction)
        #i.e if the right out of plane direction is +z, translation1 is +x, and translation2 is +y, then the angle is about the +z axis, and the right hand rule is followed
        #i.e if the right out of plane direction is -x, translation1 is +y, and translation2 is +z, then the angle is about the +x axis, and the right hand rule is followed
        #i.e if the right out of plane direction is +x, translation1 is +y, and translation2 is +z, then the angle is about the +x axis, and the right hand rule is followed
        #center the point cloud about the origin along the out of plane direction
        out_of_plane_middle = (pc[:, perp_ind].max() + pc[:, perp_ind].min()) / 2
        pc[:, perp_ind] = pc[:, perp_ind] - out_of_plane_middle
        
        pc = self.affine2D_pc(pc, perp_ind, translation1*1000, translation2*1000, angle, invert_affine)

        right_pc = (pc[multiplier*pc[:, perp_ind] > 0])
        left_pc = (pc[multiplier*pc[:, perp_ind] < 0])
        right_pc[multiplier*right_pc[:, perp_ind] < multiplier*inter_gelslim_distance / 2, perp_ind] = multiplier*inter_gelslim_distance / 2
        left_pc[multiplier*left_pc[:, perp_ind] > -multiplier*inter_gelslim_distance / 2, perp_ind] = -multiplier*inter_gelslim_distance / 2

        right_pc[:, perp_ind] = -(right_pc[:, perp_ind] - multiplier*inter_gelslim_distance / 2)*multiplier
        left_pc[:, perp_ind] = (left_pc[:, perp_ind] + multiplier*inter_gelslim_distance / 2)*multiplier

        left_pc[:, unaligned_index] = -left_pc[:, unaligned_index]

        min_depth_L = left_pc[:, perp_ind].min()
        min_depth_R = right_pc[:, perp_ind].min()

        sample_points = torch.meshgrid(self.mm_per_pixel*(torch.arange(self.image_size[0])-self.image_size[0]/2), self.mm_per_pixel*(torch.arange(self.image_size[1])-self.image_size[1]/2))

        #reshape the sample_points to be a list of 2D points
        sample_points = torch.stack((sample_points[0].flatten(), sample_points[1].flatten()), axis=1).to(self.device)

        #DEBUG CHECKPOINT: plot the point clouds as they are generated, set to False to skip
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.scatter(right_pc[:, unaligned_index], right_pc[:, aligned_index], c=right_pc[:, perp_ind],s=0.1)
            #add colorbar
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.scatter(left_pc[:, unaligned_index], left_pc[:, aligned_index], c=left_pc[:, perp_ind],s=0.1)
            #add colorbar
            plt.colorbar()
            plt.savefig('point_clouds.png')

        right_depth = torch.from_numpy(interp.griddata(right_pc[:, [unaligned_index, aligned_index]].cpu(), right_pc[:, perp_ind].cpu(), sample_points.cpu(), method='linear')).float().to(self.device)
        left_depth = torch.from_numpy(interp.griddata(left_pc[:, [unaligned_index, aligned_index]].cpu(), left_pc[:, perp_ind].cpu(), sample_points.cpu(), method='linear')).float().to(self.device)

        #remove positive values
        right_depth[right_depth > 0] = 0
        left_depth[left_depth > 0] = 0

        #remove values lower than the minimum depth
        right_depth[right_depth < min_depth_R] = min_depth_R
        left_depth[left_depth < min_depth_L] = min_depth_L

        #DEBUG CHECKPOINT: plot the sampled depth points as they are generated, set to False to skip
        if False:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.scatter(sample_points[:,0], sample_points[:,1], c=right_depth.flatten(),s=0.1)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.scatter(sample_points[:,0], sample_points[:,1], c=left_depth.flatten(),s=0.1)
            plt.colorbar()
            plt.savefig('sampled_depth.png')
            #import pdb; pdb.set_trace()

        #construct the depth image
        right_depth_image = right_depth.reshape(self.image_size)
        left_depth_image = left_depth.reshape(self.image_size)

        #replace nans with 0
        right_depth_image[torch.isnan(right_depth_image)] = 0
        left_depth_image[torch.isnan(left_depth_image)] = 0
        
        #DEBUG CHECKPOINT: plot the depth images as they are generated, set to False to skip
        if True:
            import matplotlib.pyplot as plt
            plt.subplot(1,2,1)
            plt.imshow(right_depth_image.cpu())
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(left_depth_image.cpu())
            plt.colorbar()
            plt.savefig('depth_images.png')

        return right_depth_image, left_depth_image

    def affine2D_pc(self, pc, perp_axis, translation1, translation2, angle, invert_affine=False):
        affine_matrix = torch.Tensor([[torch.cos(angle), -torch.sin(angle), translation1],
                                   [torch.sin(angle), torch.cos(angle), translation2],
                                   [0, 0, 1]]).float().to(self.device)
        if invert_affine:
            affine_matrix = torch.linalg.inv(affine_matrix)
        non_perp_indices = [0,1,2]
        non_perp_indices.remove(perp_axis)
        pc2d = pc[:, non_perp_indices]
        pcdepth = pc[:, perp_axis]
        pc2d = torch.concatenate((pc2d, torch.ones((pc2d.shape[0], 1)).to(self.device)), axis=1)
        pc2d = torch.matmul(affine_matrix, pc2d.T).T
        new_pc = pc.clone()
        new_pc[:, non_perp_indices] = pc2d[:, :2]
        new_pc[:, perp_axis] = pcdepth
        return new_pc
        
