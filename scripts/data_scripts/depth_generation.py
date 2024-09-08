from gelslim_depth.mesh_utils.depth_from_mesh import DepthImageGenerator

if __name__ == '__main__':
    mesh_dir = 'mesh'
    pc_scale = 1000 #converts from meters to mm
    dataset_dir = 'data'
    inter_gelslim_distance = 35
    gelslim_plane = '+y+z'
    LR_flip = False
    image_size = (327,420)
    image_height_mm = 12
    depth_image_generator = DepthImageGenerator(mesh_dir, pc_scale, dataset_dir, inter_gelslim_distance, gelslim_plane, LR_flip, image_size, image_height_mm)
    depth_image_generator.generate_depth_images_v1()