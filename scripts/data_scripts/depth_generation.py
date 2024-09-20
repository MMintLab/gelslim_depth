from gelslim_depth.mesh_utils.depth_from_mesh import DepthImageGenerator

if __name__ == '__main__':
    mesh_dir = 'mesh'
    object_list = ['marble','hex_key']
    pc_scale = 1000 #converts from meters to mm: 1000
    dataset_dir = '/data/william/gelslim_depth/data/real_data/'
    grasp_widths_file = '/data/william/gelslim_depth/data/grasp_widths.txt'
    grasp_width_offset = 0.0
    gelslim_plane = '+y+z'
    LR_flip = False
    image_size = (327,420)
    image_height_mm = 12
    depth_image_generator = DepthImageGenerator(mesh_dir, object_list, pc_scale, dataset_dir, grasp_widths_file, gelslim_plane, LR_flip, image_size, image_height_mm, grasp_width_offset)
    depth_image_generator.generate_depth_images_v1()