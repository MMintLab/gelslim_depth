from gelslim_depth.mesh_utils.depth_from_mesh import DepthImageGenerator
import gelslim_depth.main_config as main_config

#this script will generate depth images from the mesh files and pt files containing corresponding in_hand_poses and tactile images
#usage: python3 scripts/data_scripts/depth_generation.py
#set all the parameters in the __main__ section below

if __name__ == '__main__':
    mesh_dir = 'mesh' #directory containing the mesh files
    object_list = ['marble','hex_key'] #list of objects to generate depth images for
    pc_scale = 1000 #converts from meters to mm: 1000
    dataset_dir = main_config.DATA_PATH+'/real_data/' #directory containing the pt files
    grasp_widths_file = main_config.DATA_PATH+'/grasp_widths.txt' #file containing the grasp widths for each object
    grasp_width_offset = 0.0 #offset to add to the grasp width
    gelslim_plane = '+y+z' #plane of tactile sensor compared to the mesh, only tested for '+y+z', meaning the 
    LR_flip = False #flip the left and right tactile images
    image_size = (327,420) #width, height of tactile image and output depth image
    image_height_mm = 12 #height of tactile image in mm
    depth_image_generator = DepthImageGenerator(mesh_dir, object_list, pc_scale, dataset_dir, grasp_widths_file, gelslim_plane, LR_flip, image_size, image_height_mm, grasp_width_offset)
    depth_image_generator.generate_depth_images_v1()