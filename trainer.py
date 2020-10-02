import os
import sys
import copy
import time
import datetime

import numpy as np

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from utils import *
from model import *


class CAETrainer:
	"""A class for training CAE model

	Attributes
	----------
	config
	device : torch device
	log_file : File
	dataloader_train : DataLoader
	dataloader_val : DataLoader
	dataloader_test : DataLoader
	dataset_train : CovidCTDataset
	dataset_val : CovidCTDataset
	dataset_test : CovidCTDataset
	cnn : instance of `Extractor`
	cae : instance of `MultiCAE`
	criterion : torch.nn.MSELoss
	optimiser : torch.optim.SGD
	scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
	"""

	def __init__(self, config):
		"""
		Parameters
		----------
		config : object of `Paramaters`
			Obeject of class `Parameters` containing all required parameters
		"""

		self.config = config

		make_folder(config.weight_path)
		make_folder(config.log_path)

		write_config_to_file(config, config.save_path)

		log_file_name = os.path.join(self.config.save_path, "cae_log.txt")
		self.log_file = open(log_file_name, "wt")

		check_for_CUDA(self)

		(
			self.dataloader_train,
			self.dataloader_val,
			self.dataloader_test,
			self.dataset_train,
			self.dataset_val,
			self.dataset_test,
		) = make_dataloader(
			batch_size=self.config.batch_size,
			dataset_type=self.config.dataset,
			data_path=self.config.data_path,
			shuffle=self.config.shuffle,
			resize=self.config.resize,
			imsize=self.config.imsize,
		)

		# Models
		self.cnn = Extractor(config.pretrain_model_name)
		self.cae = MultiCAE(config.in_dim, config.code_dim, config.n_layer)
		print(self.cnn)
		print(self.cae)

		self.criterion = nn.MSELoss()
		self.optimiser = optim.SGD(self.cae.parameters(), lr=config.lr)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
			self.optimiser, mode="max", factor=0.1, patience=config.patience, verbose=True
		)

	def extract(self, dataloader):
		"""Extract all images features using pretrained CNN model

		Parameters
		----------
		dataloader : Dataloder of PyTorch
		"""

		self.cnn.to(self.device)
		self.cnn.eval()

		all_features = []
		all_labels = []
		for inputs, labels in tqdm(dataloader, desc="Extracting features"):
			inputs = inputs.to(self.device)

			with torch.set_grad_enabled(False):
				feats = self.cnn(inputs)

			all_features.append(feats.detach().cpu().numpy())
			all_labels.append(labels.detach().cpu().numpy())
			del feats

		return np.concatenate(all_features), np.concatenate(all_labels)

	def train(self):
		"""Train CAE model"""

		# Extract features
		features, labels = self.extract(self.dataloader_train)

		# Make dataset for extracted features
		dataset = SimpleDataset(features, labels)
		dataloader = DataLoader(
			dataset,
			batch_size=self.config.batch_size,
			shuffle=self.config.shuffle,
			pin_memory=True,
		)

		# Train CAE model
		self.cae.to(self.device)

		writer = SummaryWriter(self.config.log_path)
		start_time = time.time()
		best_loss = sys.maxsize
		best_model_wts = None

		for step in range(self.config.total_step):
			print("Step {}/{}:".format(step + 1, self.config.total_step))

			self.cae.train()
			running_loss = []

			for inputs, labels in tqdm(dataloader):
				inputs = inputs.to(self.device)

				# Zero the parameter gradients
				self.optimiser.zero_grad()

				# Forward
				with torch.set_grad_enabled(True):
					preds = self.cae(inputs)
					loss = self.criterion(preds, inputs)

					# Backward + optimize
					loss.backward()
					self.optimiser.step()

				# Statistics
				running_loss.append(loss.item())

			loss = np.mean(running_loss)

			if loss < best_loss:
				print("Best train loss saved ...")
				print("Model saved ...")

				best_loss = loss
				best_model_wts = copy.deepcopy(self.cae.state_dict())
				torch.save(
					best_model_wts, "{}/best_model.pth".format(self.config.weight_path)
				)

			writer.add_scalar("CAE / Train Loss", loss, step)

			self.scheduler.step(loss)

			# Print logs
			curr_time = time.time()
			curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime(
				"%Y-%m-%d %H:%M:%S"
			)
			elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
			log = "\n[{}] : Step {} elapsed [{}], Train Loss/Best: {:.4f}/{:.4f}".format(
				curr_time_str, step + 1, elapsed, loss, best_loss
			)
			print(log)
			self.log_file.write(log)
			self.log_file.flush()

		writer.close()

	def val(self, dataloader):
		"""Validate CAE model"""

		feats_train, label_train = self.extract(self.dataloader_train)
		feats_val, label_val = self.extract(dataloader)

		# CAE model
		print("Val CAE model ...")

		dataset_train = SimpleDataset(feats_train, label_train)
		dataloader_train = DataLoader(
			dataset_train,
			batch_size=self.config.batch_size,
			shuffle=False,
			pin_memory=True,
		)

		dataset_val = SimpleDataset(feats_val, label_val)
		dataloader_val = DataLoader(
			dataset_val,
			batch_size=self.config.batch_size,
			shuffle=False,
			pin_memory=True,
		)

		self.cae.eval()

		features_train = []
		targets_train = []

		features_val = []
		targets_val = []

		with torch.no_grad():
			for inputs, labels in tqdm(dataloader_train, desc="Val CAE model - train"):
				inputs = inputs.to(self.device)
				features_train.extend(
					self.cae(inputs).detach().cpu().flatten(1).numpy()
				)
				targets_train.extend(labels.detach().cpu().numpy())

			for inputs, labels in tqdm(dataloader_val, desc="Val CAE model - val"):
				inputs = inputs.to(self.device)
				features_val.extend(self.cae(inputs).detach().cpu().flatten(1).numpy())
				targets_val.extend(labels.detach().cpu().numpy())

		cae_svm = SVC(C=1, decision_function_shape="ovo")
		cae_svm.fit(features_train, targets_train)

		cae_pred = cae_svm.predict(features_val)
		cae_f1 = f1_score(targets_val, cae_pred)

		return cae_f1

	def test(self):
		"""Test CAE model"""

		feats_train, label_train = self.extract(self.dataloader_train)
		feats_test, label_test = self.extract(self.dataloader_test)

		feats_flat_train = feats_train.reshape(len(feats_train), -1)
		feats_flat_test = feats_test.reshape(len(feats_test), -1)

		# Use SVM to make prediction for pretrained model
		print("Eval pretrained model ...")
		ori_svm = SVC(C=1, decision_function_shape="ovo")
		ori_svm.fit(feats_flat_train, label_train)

		ori_pred = ori_svm.predict(feats_flat_test)
		ori_des = ori_svm.decision_function(feats_flat_test)

		ori_f1 = f1_score(label_test, ori_pred)
		ori_acc = accuracy_score(label_test, ori_pred)
		ori_auc = roc_auc_score(label_test, ori_des)

		del ori_svm

		# s Use SVM to make prediction for CAE model
		print("Eval CAE model ...")

		dataset_train = SimpleDataset(feats_train, label_train)
		dataloader_train = DataLoader(
			dataset_train,
			batch_size=self.config.batch_size,
			shuffle=False,
			pin_memory=True,
		)

		dataset_test = SimpleDataset(feats_test, label_test)
		dataloader_test = DataLoader(
			dataset_test,
			batch_size=self.config.batch_size,
			shuffle=False,
			pin_memory=True,
		)

		self.cae.load_state_dict(
			torch.load(
				"{}/best_model.pth".format(self.config.weight_path),
				map_location=self.device,
			)
		)
		self.cae.to(self.device)
		self.cae.eval()

		features_train = []
		targets_train = []
		features_test = []
		targets_test = []

		with torch.set_grad_enabled(False):
			for inputs, labels in tqdm(dataloader_train, desc="Eval CAE model - train"):
				inputs = inputs.to(self.device)
				features_train.extend(
					self.cae(inputs).detach().cpu().flatten(1).numpy()
				)
				targets_train.extend(labels)

			for inputs, labels in tqdm(dataloader_test, desc="Eval CAE model - test"):
				inputs = inputs.to(self.device)
				features_test.extend(self.cae(inputs).detach().cpu().flatten(1).numpy())
				targets_test.extend(labels)

		cae_svm = SVC(C=1, decision_function_shape="ovo")
		cae_svm.fit(features_train, targets_train)

		cae_pred = cae_svm.predict(features_test)
		cae_des = cae_svm.decision_function(features_test)

		cae_f1 = f1_score(targets_test, cae_pred)
		cae_acc = accuracy_score(targets_test, cae_pred)
		cae_auc = roc_auc_score(targets_test, cae_des)

		log = "\n[Evaluation process] Pretrained model F1 score: {:.4f}, Accuracy score: {:.4f}, AUC score: {:.4f}\n \
			CAE model F1 score: {:.4f}, Accuracy score: {:.4f}, AUC score: {:.4f}".format(
			ori_f1, ori_acc, ori_auc, cae_f1, cae_acc, cae_auc
		)

		print(log)

		self.log_file.write(log)
		self.log_file.flush()

