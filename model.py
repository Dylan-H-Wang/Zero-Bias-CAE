import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import torchvision.models as models


class CAE(nn.Module):
	"""A class for single layer concolutional autoencoder

	Attributes
	----------
	in_dim
	code_dim
	encoder : Conv2d
	relu1 : ReLU
	pool1 : MaxPool2d
	decoder : ConvTranspose2d
	relu2 : ReLU
	pool2 : MaxUnpool2d
	"""

	def __init__(self, in_dim, code_dim):
		"""
		Parameters
		----------
		in_dim : int
			Input features of CAE.
		code_dim : int
			Code features of CAE.
		"""

		super(CAE, self).__init__()

		self.in_dim = in_dim
		self.code_dim = code_dim

		self.encoder = nn.Conv2d(
			in_channels=in_dim,
			out_channels=code_dim,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		init.xavier_normal_(self.encoder.weight)
		init.constant_(self.encoder.bias, 0)  # Init bias with 0

		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, return_indices=True)

		# Init decoder weights with fliped transposed encoder weights
		trans_weights = self.filp_trans_weight(self.encoder.weight)

		self.decoder = nn.ConvTranspose2d(
			in_channels=code_dim,
			out_channels=in_dim,
			kernel_size=3,
			stride=1,
			padding=1,
		)
		self.decoder.weight.data = trans_weights.data
		init.constant_(self.decoder.bias, 0)

		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxUnpool2d(kernel_size=2)

	def forward(self, x):
		x = self.encoder(x)
		x = self.relu1(x)
		x, indices = self.pool1(x)

		x = self.pool2(x, indices)
		x = self.decoder(x)
		x = self.relu2(x)

		return x

	# Expected input: N x C x H x W
	def filp_trans_weight(self, weight):
		trans_weights = torch.transpose(weight, 2, 3)
		flip_weights = torch.flip(trans_weights, [2, 3])

		return flip_weights


class MultiCAE(nn.Module):
	"""A class for multiple layers concolutional autoencoder

	Attributes
	----------
	in_dim
	code_dim
	n_layer
	multi_cae : ModuleList
		COnsists of single layer of CAEs.
	"""

	def __init__(self, in_dim, code_dim, n_layer):
		"""
		Parameters
		----------
		in_dim : int
			Input features of CAE.
		code_dim : int
			Code features of CAE.
		n_layer : int
			Number of layers for multiple layer CAE.
		"""

		super(MultiCAE, self).__init__()

		self.in_dim = in_dim
		self.code_dim = code_dim
		self.n_layer = n_layer

		# Multi-layer CAE
		self.multi_cae = nn.ModuleList([CAE(in_dim, code_dim) for i in range(n_layer)])

	def forward(self, x):
		for cae in self.multi_cae:
			x = cae(x)
		return x

class AlexNet(models.AlexNet):
	"""AlexNet from Pytorch with removed last linear layer (classifier)"""

	def __init__(self, pretrained=False, progress=True, **kwargs):
		super().__init__(**kwargs)

		model_url = 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'
		if pretrained:
			self.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=progress))

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		return x
		
class Extractor(nn.Module):
	"""Pretrained CNN models"""

	def __init__(self, name="alexnet", freeze=True):
		super(Extractor, self).__init__()

		self.avail_model = ["alexnet"]
		assert (
			name in self.avail_model
		), "Specified model not supported, try: {}".format(self.avail_model)

		self.name = name
		self.freeze = freeze

		if self.name == "alexnet":
			self.extractor = AlexNet(pretrained=True)

		if self.freeze: 
			# Freeze extractor
			self.extractor.eval()
			for param in self.extractor.parameters():
				param.requires_grad = False

	def forward(self, x):
		x = self.extractor(x)
		return x

