from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
import concurrent.futures
from gelslim_depth.processing_utils.image_utils import get_difference_image, sample_multi_channel_image_to_desired_size, blur_depth_images, sample_multi_channel_image_to_desired_size
from gelslim_depth.processing_utils.normalization_utils import normalize_tactile_image, normalize_depth_image

class GeneralDataset(Dataset):
	def __init__(self, directory=None, pt_file_list=None, extra_directory=None, extra_pt_list=None, use_difference_image=False, depth_normalization_method='min_max_to_0_-1', image_normalization_method='mean_std',
			  separate_fingers=True, downsample_factor: int = 0.5, depth_image_blur_kernel: int = 1, depth_normalization_parameters = None, image_normalization_parameters = None, norm_scale= None, max_datapoints_per_object=None,
			  device=None, interp_method=None) -> None:
		
		assert os.path.exists(directory), f"Dataset path {directory} does not exist"

		self.parallelize = False

		self.use_difference_image = use_difference_image

		self.downsample_factor = downsample_factor

		self.depth_image_blur_kernel = depth_image_blur_kernel

		self.dataset_path = directory

		self.pt_file_list = pt_file_list

		self.extra_directory = extra_directory

		self.extra_pt_list = extra_pt_list

		self.max_datapoints_per_object = max_datapoints_per_object

		self.separate_fingers = separate_fingers

		self.device = device

		self.interp_method = interp_method

		self.entire_dataset = self.load_entire_dataset()

		self.depth_normalization_method = depth_normalization_method

		self.image_normalization_method = image_normalization_method

		self.input_tactile_image_size = (self.entire_dataset['tactile_image'][0,...].shape[1], self.entire_dataset['tactile_image'][0,...].shape[2])
		if depth_normalization_parameters is None:
			self.depth_normalization_parameters = self.calculate_depth_normalization_params()
		else:
			self.depth_normalization_parameters = depth_normalization_parameters
		if image_normalization_parameters is None:
			self.image_normalization_parameters = self.calculate_image_normalization_params()
		else:
			self.image_normalization_parameters = image_normalization_parameters
		self.norm_scale = norm_scale

	def load_object_dataset(self, object_index):
		pt_file = self.pt_file_list[object_index]

		data = torch.load(os.path.join(self.dataset_path, pt_file), map_location='cpu')

		if self.input_tactile_image_size is None:
			self.input_tactile_image_size = (int(data['tactile_image'].shape[2]*self.downsample_factor), int(data['tactile_image'].shape[3]*self.downsample_factor))

		if self.separate_fingers:
			if self.use_difference_image:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(torch.cat((get_difference_image(data['tactile_image'][:, 0:3, ...],data['base_tactile_image'][:, 0:3, ...]), get_difference_image(data['tactile_image'][:, 3:6, ...],data['base_tactile_image'][:, 3:6, ...])), dim=0), self.input_tactile_image_size, self.interp_method)
			else:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(torch.cat((data['tactile_image'][:, 0:3, ...], data['tactile_image'][:, 3:6, ...]), dim=0), self.input_tactile_image_size, self.interp_method)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(sample_multi_channel_image_to_desired_size(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.input_tactile_image_size, self.interp_method), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = sample_multi_channel_image_to_desired_size(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.input_tactile_image_size, self.interp_method)
		else:
			if self.use_difference_image:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(get_difference_image(data['tactile_image'],data['base_tactile_image']), self.input_tactile_image_size, self.interp_method)
			else:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(data['tactile_image'], self.input_tactile_image_size, self.interp_method)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(sample_multi_channel_image_to_desired_size(data['depth_image'], self.input_tactile_image_size, self.interp_method), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = sample_multi_channel_image_to_desired_size(data['depth_image'], self.input_tactile_image_size, self.interp_method)
		
		data['object_index'] = torch.tensor([object_index]*data['tactile_image'].shape[0])

		#if max_datapoints_per_object is not None, then only take a random subset of the data
		if self.max_datapoints_per_object is not None and data['tactile_image'].shape[0] > self.max_datapoints_per_object:
			indices = torch.randperm(data['tactile_image'].shape[0])
			indices = indices[:self.max_datapoints_per_object]
			data['tactile_image'] = data['tactile_image'][indices,...]
			data['depth_image'] = data['depth_image'][indices,...]
			data['object_index'] = data['object_index'][indices,...]
		return data

	def load_extra_object_dataset(self, object_index):
		pt_file = self.extra_pt_list[object_index]

		data = torch.load(os.path.join(self.extra_directory, pt_file), map_location='cpu')

		if self.separate_fingers:
			if self.use_difference_image:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(torch.cat(((data['tactile_image'][:, 0:3, ...]-data['base_tactile_image'][:, 0:3, ...])/2.0+127.5, (data['tactile_image'][:, 3:6, ...]-data['base_tactile_image'][:, 3:6, ...])/2.0+127.5), dim=0), self.input_tactile_image_size, self.interp_method)
			else:
				data['tactile_image'] = sample_multi_channel_image_to_desired_size(torch.cat((data['tactile_image'][:, 0:3, ...], data['tactile_image'][:, 3:6, ...]), dim=0), self.input_tactile_image_size, self.interp_method)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(sample_multi_channel_image_to_desired_size(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.input_tactile_image_size, self.interp_method), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = sample_multi_channel_image_to_desired_size(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.input_tactile_image_size, self.interp_method)
		else:
			data['tactile_image'] = sample_multi_channel_image_to_desired_size(data['tactile_image'], self.input_tactile_image_size, self.interp_method)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(sample_multi_channel_image_to_desired_size(data['depth_image'], self.input_tactile_image_size, self.interp_method), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = sample_multi_channel_image_to_desired_size(data['depth_image'], self.input_tactile_image_size, self.interp_method)
		
		data['object_index'] = torch.tensor([object_index]*data['tactile_image'].shape[0])

		#if max_datapoints_per_object is not None, then only take a random subset of the data
		if self.max_datapoints_per_object is not None and data['tactile_image'].shape[0] > self.max_datapoints_per_object:
			indices = torch.randperm(data['tactile_image'].shape[0])
			indices = indices[:self.max_datapoints_per_object]
			data['tactile_image'] = data['tactile_image'][indices,...]
			data['depth_image'] = data['depth_image'][indices,...]
			data['object_index'] = data['object_index'][indices,...]
		return data
	
	def load_entire_dataset(self):
		entire_dataset = {}
		#number_of_objects is the number of keys in the filename dict
		number_of_objects = len(self.pt_file_list)
		
		if self.parallelize:
			print('loading main dataset in parallel')
			with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
				futures = [executor.submit(self.load_object_dataset, object_index) for object_index in range(number_of_objects)]

				for future in tqdm(concurrent.futures.as_completed(futures), total=number_of_objects):
					object_dataset = future.result()
					for key, data in object_dataset.items():
						if key in entire_dataset:
							entire_dataset[key] = torch.cat((entire_dataset[key], data), dim=0)
						else:
							entire_dataset[key] = data
			if self.extra_directory is not None:
				print('loading extra dataset in parallel')
				number_of_extra_objects = len(self.extra_pt_list)
				with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
					futures = [executor.submit(self.load_extra_object_dataset, object_index) for object_index in range(number_of_extra_objects)]

					for future in tqdm(concurrent.futures.as_completed(futures), total=number_of_extra_objects):
						object_dataset = future.result()
						for key, data in object_dataset.items():
							if key in entire_dataset:
								entire_dataset[key] = torch.cat((entire_dataset[key], data), dim=0)
							else:
								entire_dataset[key] = data
		else:
			print('loading main dataset sequentially')
			for object_index in tqdm(range(number_of_objects)):
				object_dataset = self.load_object_dataset(object_index)
				for key, data in object_dataset.items():
					if key in entire_dataset:
						entire_dataset[key] = torch.cat((entire_dataset[key], data), dim=0)
					else:
						entire_dataset[key] = data
			if self.extra_directory is not None:
				print('loading extra dataset sequentially')
				number_of_extra_objects = len(self.extra_pt_list)
				for object_index in tqdm(range(number_of_extra_objects)):
					object_dataset = self.load_extra_object_dataset(object_index)
					for key, data in object_dataset.items():
						if key in entire_dataset:
							entire_dataset[key] = torch.cat((entire_dataset[key], data), dim=0)
						else:
							entire_dataset[key] = data
		return entire_dataset
	
	def blur_depth_images(self, depth, depth_image_blur_kernel):
		depth = blur_depth_images(depth, depth_image_blur_kernel)
		return depth
	
	def calculate_depth_normalization_params(self):
		max_depth = self.entire_dataset['depth_image'].max().item()
		min_depth = self.entire_dataset['depth_image'].min().item()
		mean_depth = self.entire_dataset['depth_image'].mean().item()
		std_depth = self.entire_dataset['depth_image'].std().item()
		return (min_depth, max_depth, mean_depth, std_depth)
	
	def calculate_image_normalization_params(self):
		#get parameters for each channel
		n_channels = self.entire_dataset['tactile_image'].shape[1]
		means = []
		stds = []
		maxes = []
		mins = []
		for i in range(n_channels):
			channel = self.entire_dataset['tactile_image'][:,i,...]
			maxes.append(channel.max().item())
			mins.append(channel.min().item())
			means.append(channel.mean().item())
			stds.append(channel.std().item())
		return (mins, maxes, means, stds)
	
	def normalize_sample(self, sample):
		new_sample = {}
		new_sample['tactile_image'] = normalize_tactile_image(sample['tactile_image'], self.image_normalization_method, self.norm_scale, self.image_normalization_parameters)
		new_sample['depth_image'] = normalize_depth_image(sample['depth_image'], self.depth_normalization_method, self.norm_scale, self.depth_normalization_parameters)
		return new_sample
	
	def __len__(self):
		# return the number of total samples contained in the dataset
		return self.entire_dataset['tactile_image'].shape[0]
	def __getitem__(self, idx):
		# get the sample at the given index
		tactile_image = self.entire_dataset['tactile_image'][idx,...]
		depth_image = self.entire_dataset['depth_image'][idx,...]
		object_index = self.entire_dataset['object_index'][idx]
		#normalize the in_hand_pose
		sample = {}
		sample['tactile_image'] = tactile_image
		sample['depth_image'] = depth_image
		sample = self.normalize_sample(sample, self.depth_normalization_parameters)
		sample['object_index'] = object_index
		return sample