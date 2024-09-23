from gelslim_depth.processing_utils.normalization_utils import denormalize_depth_image, normalize_tactile_image
from gelslim_depth.processing_utils.image_utils import sample_multi_channel_image_to_desired_size

def predict_depth_from_RGB(images, model, output_size, config):
    images = sample_multi_channel_image_to_desired_size(images, config.input_tactile_image_size, config.interp_method)
    images = normalize_tactile_image(images, config.tactile_normalization_method, config.norm_scale, config.tactile_normalization_parameters)
    depth = model(x=images)
    depth = denormalize_depth_image(depth, config.depth_normalization_method, config.norm_scale, config.depth_normalization_parameters)
    depth = sample_multi_channel_image_to_desired_size(depth, output_size, config.interp_method)
    return depth