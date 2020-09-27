import os

import torch
from torch.utils.data.dataset import Dataset

from torchvision import transforms
from torchvision import datasets as dset

from PIL import Image


def make_folder(path):
	"""Make folders if the given path does not exist.

	If the argument `path` does not exist, make directories for it including 
	all intermediate-level directories needed to contain the leaf directory.

	Parameters
	----------
	path : str
		Path of the folder
	"""

	if not os.path.exists(path):
		os.makedirs(path)


def read_txt(path):
	"""Read all lines from .txt files.

	Parameters
	----------
	path : str
		Path of the file

	Returns
	-------
	txt_data : list if str
		All lines in the text file
	"""

	with open(path) as f:
		lines = f.readlines()

	txt_data = [line.strip() for line in lines]
	return txt_data


def write_config_to_file(config, save_path):
	"""Record and save current parameter settings

	Parameters
	----------
	config : object of class `Parameters`
		Object of class `Parameters`
	save_path : str
		Path to save the file
	"""

	with open(os.path.join(save_path, "config.txt"), "w") as file:
		for arg in vars(config):
			file.write(str(arg) + ": " + str(getattr(config, arg)) + "\n")


def check_for_CUDA(target):
	"""Check whether CUDA is available

	Parameters
	----------
	target : object
		An object which contains attribute `config`
	"""

	if target.config.cuda and torch.cuda.is_available():
		print("CUDA is available!")
		target.device = torch.device("cuda:0")
		print("Using {}".format(target.device))
	else:
		print("Cuda is NOT available, running on CPU.")
		target.device = torch.device("cpu")

	if torch.cuda.is_available() and not target.config.cuda:
		print("WARNING: You have a CUDA device")


def make_transform(resize=True, imsize=227, totensor=True):
	"""Make transformer for dataset which can be used in torchvision

	Parameters
	----------
	resize : bool, optional
		Whether to resize the image. Default value is True.
	imsize : int, optional
		The dimension of the image. Default value is 227.
	totensor : bool, optional
		Whether to transform image into tensors. Default value is True.

	Returns
	-------
	transform : object of torchvision.transforms
		Composed transformation functions.
	"""

	options = []
	if resize:
		options.append(transforms.Resize((imsize, imsize)))
	if totensor:
		options.append(transforms.ToTensor())
	transform = transforms.Compose(options)
	return transform


def make_dataloader(
	batch_size,
	dataset_type,
	data_path,
	shuffle=True,
	resize=True,
	imsize=227,
	totensor=True,
):
	"""Make transformer for dataset which can be used in torchvision

	Parameters
	----------
	batch_size : int
		Batch size of samples.
	dataset_type : str
		Type of dataset to use.
	data_path : str
		Path of dataset.
	shuffle : bool, optional
		Whether to shuffle the dataset. Default value is True.
	resize : bool, optional
		Whether to resize the image. Default value is True.
	imsize : int, optional
		The dimension of the image. Default value is 227.
	totensor : bool, optional
		Whether to transform image into tensors. Default value is True.

	Returns
	-------
	dataloader_train : DataLoader of PyTorch
		Dataloader for training set.
	dataloader_val : DataLoader of PyTorch
		Dataloader for validation set.
	dataloader_test : DataLoader of PyTorch
		Dataloader for testing set.
	dataset_train : CovidCTDataset of PyTorch
		Dataset for training set.
	dataset_val : CovidCTDataset of PyTorch
		Dataset for validation set.
	dataset_test : CovidCTDataset of PyTorch
		Dataset for testing set.
	"""

	avail_type = ["cifar10", "covid-ct"]
	assert (
		dataset_type in avail_type
	), "Specified dataset not supported, try: {}".format(avail_type)

	# Make transform
	transform = make_transform(resize=resize, imsize=imsize, totensor=totensor)

	# Make dataset
	if dataset_type == "cifar10":
		if not os.path.exists(data_path):
			print(
				"data_path does not exist! Given: {}\nDownloading CIFAR10 dataset...".format(
					data_path
				)
			)
		dataset_train = dset.CIFAR10(
			root=data_path, train=True, download=True, transform=transform
		)
		dataset_test = dset.CIFAR10(
			root=data_path, train=False, download=True, transform=transform
		)

	elif dataset_type == "covid-ct":
		assert os.path.exists(data_path), (
			"data_path does not exist! Given: " + data_path
		)

		train_path_covid = os.path.join(
			data_path, "Data-split", "COVID", "trainCT_COVID.txt"
		)
		val_path_covid = os.path.join(
			data_path, "Data-split", "COVID", "valCT_COVID.txt"
		)
		test_path_covid = os.path.join(
			data_path, "Data-split", "COVID", "testCT_COVID.txt"
		)

		train_path_noncovid = os.path.join(
			data_path, "Data-split", "NonCOVID", "trainCT_NonCOVID.txt"
		)
		val_path_noncovid = os.path.join(
			data_path, "Data-split", "NonCOVID", "valCT_NonCOVID.txt"
		)
		test_path_noncovid = os.path.join(
			data_path, "Data-split", "NonCOVID", "testCT_NonCOVID.txt"
		)

		dataset_train = CovidCTDataset(
			root=data_path,
			split_COVID=train_path_covid,
			split_NonCOVID=train_path_noncovid,
			transform=transform,
		)
		dataset_val = CovidCTDataset(
			root=data_path,
			split_COVID=val_path_covid,
			split_NonCOVID=val_path_noncovid,
			transform=transform,
		)
		dataset_test = CovidCTDataset(
			root=data_path,
			split_COVID=test_path_covid,
			split_NonCOVID=test_path_noncovid,
			transform=transform,
		)

	# Make dataloader from dataset
	dataloader_train = torch.utils.data.DataLoader(
		dataset_train, batch_size=batch_size, shuffle=shuffle, pin_memory=True,
	)
	dataloader_val = torch.utils.data.DataLoader(
		dataset_val, batch_size=batch_size, shuffle=False, pin_memory=True,
	)
	dataloader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=batch_size, shuffle=False, pin_memory=True,
	)

	print(
		"Dataset is split with train: {}, val: {}, test: {}".format(
			len(dataset_train), len(dataset_val), len(dataset_test)
		)
	)

	return (
		dataloader_train,
		dataloader_val,
		dataloader_test,
		dataset_train,
		dataset_val,
		dataset_test,
	)


class SimpleDataset(Dataset):
	"""Convert give data to Pytorch dataset

	Attributs
	----------
	data
	label
	"""

	def __init__(self, data, label):
		self.data = data
		self.label = label

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.label[idx]


class CovidCTDataset(Dataset):
	"""Adapted from COVID-CT baseline. https://github.com/UCSD-AI4H/COVID-CT.

	"""

	def __init__(self, root, split_COVID, split_NonCOVID, transform=None):
		self.root = root
		self.txt_path = [split_COVID, split_NonCOVID]
		self.classes = ["CT_COVID", "CT_NonCOVID"]
		self.num_cls = len(self.classes)
		self.img_list = []

		for c in range(self.num_cls):
			cls_list = [
				[os.path.join(self.root, self.classes[c], item), c]
				for item in read_txt(self.txt_path[c])
			]
			self.img_list += cls_list
		self.transform = transform
		self.normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
		)

	def __len__(self):
		return len(self.img_list)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_path = self.img_list[idx][0]
		image = Image.open(img_path).convert("RGB")

		if self.transform:
			image = self.transform(image)
			image = self.normalize(image)

		return image, int(self.img_list[idx][1])

