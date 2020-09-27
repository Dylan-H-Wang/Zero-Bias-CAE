import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models

import numpy as np

from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from utils import *


class AlexNet:
	def __init__(self):
		self.device = torch.device("cuda:0")
		self.total_step = 150
		self.lr = 1e-3

		self.model = models.alexnet(pretrained=True)
		self.model.classifier[-1] = nn.Linear(4096, 2)
		self.model.to(self.device)

		self.criterion = nn.CrossEntropyLoss()
		self.optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

		(
			self.dataloader_train,
			self.dataloader_val,
			self.dataloader_test,
			self.dataset_train,
			self.dataset_val,
			self.dataset_test,
		) = make_dataloader(
			batch_size=256,
			dataset_type="covid-ct",
			data_path="../data/COVID-CT",
			shuffle=True,
			resize=True,
			imsize=227,
		)

	def train(self):
		start_time = time.time()
		best_f1 = -1
		best_acc = -1
		best_loss = 100
		best_model_wts = None

		for step in range(self.total_step):
			print("Step {}/{}:".format(step + 1, self.total_step))

			self.model.train()
			running_loss = []

			for inputs, labels in tqdm(self.dataloader_train):
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)

				# Zero the parameter gradients
				self.optimiser.zero_grad()

				# Forward
				with torch.set_grad_enabled(True):
					preds = self.model(inputs)
					loss = self.criterion(preds, labels)

					# Backward + optimize
					loss.backward()
					self.optimiser.step()

				# Statistics
				running_loss.append(loss.item())

			loss = np.mean(running_loss)

			# Validation
			loss_val, f1_val, acc_val = self.val()

			# Save model
			if f1_val > best_f1:
				print("Best f1 saved ...")
				best_f1 = f1_val
				print("Best model saved ...")
				best_model_wts = self.model.state_dict()

			if acc_val > best_acc:
				print("Best accuracy saved ...")
				best_acc = acc_val

			if loss < best_loss:
				print("Best loss saved ...")
				best_loss = loss

			# Print logs
			curr_time = time.time()
			curr_time_str = datetime.datetime.fromtimestamp(curr_time).strftime(
				"%Y-%m-%d %H:%M:%S"
			)
			elapsed = str(datetime.timedelta(seconds=(curr_time - start_time)))
			log = "\n[{}] : Step {} elapsed [{}], Train Loss/Best: {:.4f}/{:.4f}, Val Loss: {:.4f}, F1_score/Best: {:.4f}/{:.4f}, Accuracy/Best: {:.4f}/{:.4f}".format(
				curr_time_str,
				step + 1,
				elapsed,
				loss,
				best_loss,
				loss_val,
				f1_val,
				best_f1,
				acc_val,
				best_acc,
			)
			print(log)

		self.model.load_state_dict(best_model_wts)

	def val(self):
		self.model.eval()

		running_loss = []
		all_preds = []
		all_labels = []

		for inputs, labels in tqdm(self.dataloader_val):
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			# Forward
			with torch.no_grad():
				preds = self.model(inputs)
				loss = self.criterion(preds, labels)
				preds = preds.argmax(dim=1)

			# Statistics
			running_loss.append(loss.item())
			all_preds.extend(preds.detach().cpu().numpy())
			all_labels.extend(labels.detach().cpu().numpy())

		loss = np.mean(running_loss)
		f1 = f1_score(all_labels, all_preds)
		acc = accuracy_score(all_labels, all_preds)

		return loss, f1, acc

	def test(self):
		self.model.eval()

		running_loss = []
		all_preds = []
		all_labels = []

		for inputs, labels in tqdm(self.dataloader_test):
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)

			# Forward
			with torch.no_grad():
				preds = self.model(inputs)
				loss = self.criterion(preds, labels)
				preds = preds.argmax(dim=1)

			# Statistics
			running_loss.append(loss.item())
			all_preds.extend(preds.detach().cpu().numpy())
			all_labels.extend(labels.detach().cpu().numpy())

		loss = np.mean(running_loss)
		f1 = f1_score(all_labels, all_preds)
		acc = accuracy_score(all_labels, all_preds)

		print(
			"[Evaluation process] AlexNet model Loss: {:.4f}, F1 score: {:.4f}, Accuracy score: {:.4f}".format(
				loss, f1, acc
			)
		)