import os
import datetime


class Parameters:
	"""A class used to setup parameters

	All parameters that can be finetuned are aggregated in this class.

	Attributes
	----------
	dataset
	data_path
	save_path
	pretrain_model_name
	batch_size
	total_step
	lr
	in_dim
	code_dim
	n_layer
	patience
	shuffle
	resize
	imsize
	cuda
	name
	weight_path : str
		Path to save model weight files.
	log_path : str
		Path to save tensorboard log files.
	"""

	def __init__(
		self,
		dataset,
		data_path,
		save_path="./logs",
		pretrain_model_name="alexnet",
		batch_size=16,
		total_step=150,
		lr=0.001,
		in_dim=256,
		code_dim=4096,
		n_layer=2,
		patience=100,
		shuffle=True,
		resize=True,
		imsize=227,
		cuda=True,
		name="zero_bias_cae_alexnet",
	):
		"""
		Parameters
		----------
			dataset : str
				Type of dataset to use
			data_path : str
				Path of dataset
			save_path : str, optional
				Path where to save log files. Default log path is "./logs"
			pretrain_model_name : str, optional
				Pretrained CNN model name. Default pretrained model will be Alexnet.
			batch_size : int, optional
				Batch size of training sample. Default batch size is 16.
			total_step : int, optional
				Total steps of training. Default steps are 150.
			lr : float, optional
				Learning rate. Default learning rate is 1e-3.
			in_dim : int, optional
				Input features of CAE. Default value is 256.
			code_dim : int, optional
				Code features of CAE. Default value is 4096.
			n_layer : int, optional
				Number of layers of CAE. Default value is 2.
			patience : int, optional
				Patience for learning rate scheduler.
			shuffle : bool, optional
				Whether to shuffle the dataset. Default value is True.
			resize : bool, optional
				Whether to resize the dataset image. Default value is True.
			imsize : int, optional
				The dimension of image. Default value is 227.
			cuda : bool, optional
				Whether to use CUDA. Default value is True.
			name : str, option
				Name for the log files. Default value is "zero_bias_cae_alexnet"/
					  
		"""

		self.dataset = dataset
		self.data_path = data_path
		self.save_path = save_path
		self.pretrain_model_name = pretrain_model_name
		self.batch_size = batch_size
		self.total_step = total_step
		self.lr = lr
		self.in_dim = in_dim
		self.code_dim = code_dim
		self.n_layer = n_layer
		self.resize = resize
		self.patience = patience
		self.shuffle = shuffle
		self.imsize = imsize
		self.cuda = cuda
		self.name = name

		self.name = "{0}_{1}_{2:%Y%m%d_%H%M%S}".format(
			self.name, os.path.basename(self.dataset), datetime.datetime.now()
		)
		self.save_path = os.path.join(self.save_path, self.name)
		self.weight_path = os.path.join(self.save_path, "weights")
		self.log_path = os.path.join(self.save_path, "logs")

