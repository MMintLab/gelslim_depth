# The GelSlim 4.0 Depth Estimation Package
Neural-network-based depth estimation from RGB vision-based tactile sensor GelSlim 4.0 <br />
![GIF of Grasping a Small Screw with Depth Estimation](https://github.com/MMintLab/gelslim_depth/blob/master/media/animations/small_screw.gif?raw=true)
![GIF of Grasping a Fin Ray Finger with Depth Estimation](https://github.com/MMintLab/gelslim_depth/blob/master/media/animations/finray.gif?raw=true)<br />
![GIF of Grasping a Hex Key with Depth Estimation](https://github.com/MMintLab/gelslim_depth/blob/master/media/animations/hex_key.gif?raw=true)
![GIF of Grasping a Pointed Connector with Depth Estimation](https://github.com/MMintLab/gelslim_depth/blob/master/media/animations/pointed_connector.gif?raw=true)

Tested On: <br />
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.0+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.8+-blue.svg?logo=python&style=for-the-badge" /></a>

## Installation

1. [Install PyTorch](https://pytorch.org/get-started/locally/)

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Clone `gelslim_depth` with git to create the root directory

4. Install `gelslim_depth` (run in `gelslim_depth` root directory)
```bash
pip install -e .
```

## Dataset
Alongside this package we provide a compatible [dataset](https://www.dropbox.com/scl/fo/lo5j63dsbl2ybmmq55auv/ALiCfoR2eY9l1bpKLpuUnZQ?rlkey=0ktcfctv66mk8ftguaxazfsox&st=6gg10fe1&dl=0) consisting of `.pt` PyTorch tensor dictionaries for quick data loading. These are organized into directories, pre-split for training, validation, and testing purposes.
- `train_data`: Very large `n_samples` (1000–3000) dictionaries, informs the training of the depth estimator network
- `validation_data`: Smaller `n_samples` (100–300) dictionaries, informs the selection of the best network weights in training output
- `test_data`: Smaller `n_samples` (100–300) dictionaries, unseen during training, but still available for comparison with ground truth
- `real_data`: Even smaller `n_samples` (40–100) dictionaries, which may or may not have paired depth images or corresponding mesh files

All dictionaries have the following keys relevant to this package (other keys may exist, ignore them) with all keys aligned along the `n_samples` dimension where applicable.
- `tactile_image`: A `n_samples x 6 x 320 x 427` tensor containing `n_samples` RGB GelSlim images. The 6 channels correspond to Left GelSlim RGB followed by Right GelSlim RGB.
- `depth_images`: A `n_samples x 2 x 320 x 427` tensor containing `n_samples` depth images with each pixel value as distance from the camera in mm, where the undeformed image is 0. This means all pixel values should be negative. These are the 'ground truth' used in training and were generated using `scripts/data_scripts/generate_depth.py`. The two channels correspond to Left Depth and Right Depth.
- `in_hand_pose`: A `n_samples x 3` tensor containing `n_samples` in_hand_poses, the SE(2) or planar pose at which the object frame (the coordinate frame for the corresponding mesh file) sits in the grasp_frame, with the form (y (mm), z(mm), theta(rad))
- `base_tactile_image`: A `1 x 6 x 320 x 427` or `n_samples x 6 x 320 x 427` tensor representing the undeformed image. Some of the datasets were collected with just one undeformed image and some were collected with one for each new grasp.
- `grasp_widths`: A `n_samples` tensor listing the grasp widths (in mm) if this was recorded for the dataset. This is used with the mesh file to generate the ground truth depth images. For objects where this was not recorded, a single grasp width to be used for all data points is listed in `grasp_widths.txt`.

Also with this dataset are some txt files for configuration:
- `test_objects.txt`: Objects listed here will be entirely left out of the training and validation sets
- `validation_objects.txt`: Objects listed here will be entirely left out of the training and testing sets
The following txt configuration files are only for object dataset dictionaries which include the ground truth depth image generated from the mesh
- `real_data/train_real_objects.txt`: Place names of objects in `real_data` here if you want them included in the training set, we found this to assist training even though there are less data points for these objects.
- `real_data/validation_real_objects.txt`: Place names of objects in `real_data` here if you want them included in the validation set
- `real_data/test_real_objects.txt`: Place names of objects in `real_data` here if you want them included in the test set

IMPORTANT: After downloading this data, update `main_config.py` with its absolute path on your system:
```
DATA_PATH = '/path/to/GelSlim4.0DepthDataset'
```

You can add your own data from your own GelSlim 4.0! As long as it exists in this same format. To view the paired data in these `.pt` files, run `view_pt.py` as follows:

```bash
python scripts/data_scripts/view_pt.py <sub_dir> <data_name>
```

For example:
```bash
python scripts/data_scripts/view_pt.py test_data pattern_31_rod_test.pt
```

## New Dataset Generation
If you wish to regenerate the ground truth depth images for our dataset or generate your own for new objects with in hand pose-labeled GelSlim 4.0 images, you may run (from the `gelslim_depth` root directory):
```bash
python scripts/data_scripts/depth_generation.py
```
This script can be configured directly in the file, including the folder in which to generate the depth images, and the list of objects for which you'd like to generate the depth images. You must have a corresponding STL file in `gelslim_depth/mesh`.

If you have data in the form of `.pt` files that are not split into train, validation, and test sets, place them in `DATA_PATH` (outside of the `train_data`, etc. folders) and run `split_data.py` as follows:

```bash
python scripts/data_scripts/split_data.py <device>
```

For example:
```bash
python scripts/data_scripts/split_data.py cuda:0
```
This will perform the segmentation of all `.pt` files in `DATA_PATH` into train, test, and validation folders. This will by default delete the original `.pt` file due to the large file sizes. This segmentation will take place on `<device>` as recognized by PyTorch.

## Training U-Nets
We provide the [U-Net](https://arxiv.org/abs/1505.04597) architecture for depth estimation with customizeable layers and hyperparameters in `gelslim_depth/gelslim_depth/models/unet.py`. Our dataset class is in `gelslim_depth/gelslim_depth/datasets/general_dataset.py`. You'll find that this and other scripts rely on the code within `gelslim_depth/gelslim_depth/processing_utils`.

To run a training of the U-Net, run `train_unet.py` as follows:

```bash
python train_utils/train_unet.py <weights_name> <gpu> --exclude_objects [str: objects to exclude] --activation_func [relu, tanh, mish] --max_datapoints_per_object [int]
```
With additional arguments:
```
--train_indefinitely: do not stop the training when a rising validation loss is detected (lowest validation loss is still the default saved checkpoint)
--use_difference_image: use the difference between the deformed and undeformed tactile images as the input to the U-Net (recommended)
```
For example:
```bash
python train_utils/train_unet.py unet_model_1 1 --exclude_objects peg1 peg2 --activation_func relu --max_datapoints_per_object 1000 --train_indefinitely --use_difference_image
```

Each epoch of training may take a while given the data size and network size, but it will likely converge to a usable network in less than 100 epochs. This script will create and populate the following directories within ```gelslim_depth/train_output```:
- `live_display`: The latest (not necessarily best/currently saved) checkpoint's performance visualized on the datasets. This plotting can be disabled by setting `num_images_to_display_live = 0` in `train_unet.py`
- `weights`: The model checkpoints
- `loss_curves`: Live plotted loss curves
- `loss_values`: The logged output of the training, validation, and test losses

This will also populate `gelslim_depth/gelslim_depth/config` with an importable `.py` config file unique to the training parameters that were set, this is paired with the weights checkpoint file for testing the resulting model and downstream tasks.

## Testing U-Nets
To test a trained model, use ```test_depth_estimation.py```:

```bash
python test_utils/test_depth_estimation.py <weights_name> <gpu> <sub_dir> [object1, object2, ...]
```

This will visualize using the provided weights and device the estimation using all objects in the list `[object1, object2, ...]` that lie within `<sub-dir>`. For example:

```bash
python test_utils/test_depth_estimation.py unet_bigdata 0 real_data button marble edge
```
If no objects are given after the `<sub_dir>`, all objects within `<sub_dir>` will be plotted.

## Using depth estimation in your own code
If your require our depth estimation algorithm for your downstream robotic tasks, we provide the following capabilities:

Import the processing functions:
```
from gelslim_depth.processing_utils.normalization_utils import denormalize_depth_image, normalize_tactile_image
from gelslim_depth.processing_utils.image_utils import sample_multi_channel_image_to_desired_size
```

Alternatively, import the complete prediction function which includes these:
```
from gelslim_depth.processing_utils.complete_prediction import predict_depth_from_RGB
```

This is a function which takes in arguments ```(images, model, output_size, config)```.

Also import this function, if applicable (it is recommended to use difference images for depth estimation):
```
from gelslim_depth.processing_utils.image_utils import get_difference_image
```

### An example external use of our functionality

1. In whatever way you acquire tactile images (ROS, webcam, etc.), first convert the RGB images to a torch tensor `tactile_image` with dimensions `N x 3 x H x W` depending on the size of the images that the trained model has seen.
2. Do the same for the undeformed tactile image (again, this can be `1 x 3 x H x W` and it should work) if applicable, and save as variable `base_tactile_image`.
3. Ensure these RGB images have the original `uint8` values from 0 to 255 but the tensor datatype should be float.
4. Then, the estimation can begin as such, with defined `weights_name` and `<device>`:

```
import gelslim_depth.config.config_<weights_name>.py as config
from gelslim_depth.models.unet import UNet

network = UNet(n_channels=3, n_classes=1, layer_dimensions=config.CNN_dimensions, kernel_size=config.kernel_size, maxpool_size=config.maxpool_size, upconv_stride=config.upconv_stride).to(<device>)

weights = /path/to/checkpoints/<weights_name>.pth

network.load_state_dict(torch.load(weights, map_location=device))

diff_image = get_difference_image(tactile_image, base_tactile_image)

depth_output = predict_depth_from_RGB(diff_image, network, (diff_image.shape[2],diff_image.shape[3]), config)
```

Alternatively, the weights can be imported as recorded by config, this ensures the two are compatible:

```
weights_name = <weights_name>
model.load_state_dict(torch.load(config.weights_path+weights_name+'.pth', map_location=device))
```