from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
import concurrent.futures

class GeneralDataset(Dataset):
	def __init__(self, directory=None, pt_file_list=None, extra_directory=None, extra_pt_list=None, use_difference_image=False,
			  separate_fingers=True, downsample_factor: int = 0.5, depth_image_blur_kernel: int = 1, depth_normalization_parameters = None, norm_scale= None, max_datapoints_per_object=None,
			  device=None) -> None:
		
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

		self.entire_dataset = self.load_entire_dataset()

		self.input_tactile_image_size = (self.entire_dataset['tactile_image'][0,...].shape[1], self.entire_dataset['tactile_image'][0,...].shape[2])
		if depth_normalization_parameters is None:
			self.depth_normalization_parameters = self.calculate_depth_normalization_params()
		else:
			self.depth_normalization_parameters = depth_normalization_parameters
		self.norm_scale = norm_scale

	def load_object_dataset(self, object_index):
		pt_file = self.pt_file_list[object_index]

		data = torch.load(os.path.join(self.dataset_path, pt_file), map_location='cpu')

		if self.separate_fingers:
			if self.use_difference_image:
				data['tactile_image'] = self.downsample_tactile_images(torch.cat(((data['tactile_image'][:, 0:3, ...]-data['base_tactile_image'][:, 0:3, ...])/2.0+127.5, (data['tactile_image'][:, 3:6, ...]-data['base_tactile_image'][:, 3:6, ...])/2.0+127.5), dim=0), self.downsample_factor)
			else:
				data['tactile_image'] = self.downsample_tactile_images(torch.cat((data['tactile_image'][:, 0:3, ...], data['tactile_image'][:, 3:6, ...]), dim=0), self.downsample_factor)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(self.downsample_depth_images(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.downsample_factor), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = self.downsample_depth_images(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.downsample_factor)
		else:
			data['tactile_image'] = self.downsample_tactile_images(data['tactile_image'], self.downsample_factor)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(self.downsample_depth_images(data['depth_image'], self.downsample_factor), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = self.downsample_depth_images(data['depth_image'], self.downsample_factor)
		
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
				data['tactile_image'] = self.downsample_tactile_images(torch.cat(((data['tactile_image'][:, 0:3, ...]-data['base_tactile_image'][:, 0:3, ...])/2.0+127.5, (data['tactile_image'][:, 3:6, ...]-data['base_tactile_image'][:, 3:6, ...])/2.0+127.5), dim=0), self.downsample_factor)
			else:
				data['tactile_image'] = self.downsample_tactile_images(torch.cat((data['tactile_image'][:, 0:3, ...], data['tactile_image'][:, 3:6, ...]), dim=0), self.downsample_factor)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(self.downsample_depth_images(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.downsample_factor), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = self.downsample_depth_images(torch.cat((data['depth_image'][:, 0:1, ...], data['depth_image'][:, 1:2, ...]), dim=0), self.downsample_factor)
		else:
			data['tactile_image'] = self.downsample_tactile_images(data['tactile_image'], self.downsample_factor)
			if self.depth_image_blur_kernel > 1:
				data['depth_image'] = self.blur_depth_images(self.downsample_depth_images(data['depth_image'], self.downsample_factor), self.depth_image_blur_kernel)
			else:
				data['depth_image'] = self.downsample_depth_images(data['depth_image'], self.downsample_factor)
		
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

	def downsample_tactile_images(self, tactile_images, downsample_factor):
		input_size = tactile_images.shape
		if self.separate_fingers:
			output_size = (input_size[0], 3, int(input_size[2]*downsample_factor), int(input_size[3]*downsample_factor))
		else:
			output_size = (input_size[0], 6, int(input_size[2]*downsample_factor), int(input_size[3]*downsample_factor))
		self.input_tactile_image_size = (output_size[2], output_size[3])
		tactile_images = F.interpolate(tactile_images, size=(output_size[2], output_size[3]), mode='area')
		return tactile_images

	def downsample_depth_images(self, depth, downsample_factor):
		input_size = depth.shape
		if self.separate_fingers:
			output_size = (input_size[0], 1, int(input_size[2]*downsample_factor), int(input_size[3]*downsample_factor))
		else:
			output_size = (input_size[0], 2, int(input_size[2]*downsample_factor), int(input_size[3]*downsample_factor))
		depth = F.interpolate(depth, size=self.input_tactile_image_size, mode='area')
		return depth
	
	def blur_depth_images(self, depth, depth_image_blur_kernel):
		depth = TF.gaussian_blur(depth, kernel_size=depth_image_blur_kernel)
		return depth
	
	def calculate_depth_normalization_params(self):
		max_depth = self.entire_dataset['depth_image'].max()
		min_depth = self.entire_dataset['depth_image'].min()
		return (min_depth, max_depth)
	
	def normalize_sample(self, sample, depth_normalization_parameters):
		min_depth, max_depth = depth_normalization_parameters
		middle_depth = (max_depth + min_depth)/2
		new_sample = {}
		new_sample['tactile_image'] = sample['tactile_image'].clone()/255.0
		new_sample['depth_image'] = -self.norm_scale*(sample['depth_image'] - min_depth)/(max_depth - min_depth)
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