from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms as tf
import matplotlib.pyplot as plt
import time
import copy
import os


data_transforms = {
	'train'	: tf.Compose([tf.Resize(224), tf.RandomHorizontalFlip(), tf.ToTensor()]),
	'val'	: tf.Compose([tf.Resize(224), tf.ToTensor()]),
}

dsets			= {x: datasets.ImageFolder(x, data_transforms[x]) for x in ['train', 'val']}
dset_loaders	= {x: torch.utils.data.DataLoader(dsets[x], batch_size=64, shuffle=True, num_workers=0) for x in ['train', 'val']}
dset_sizes		= {x: len(dsets[x]) for x in ['train', 'val']}

def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=25):
	since = time.time()
	best_acc = 0.0
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
	
		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				optimizer = lr_scheduler(optimizer, epoch)
				model.train(True)  # Set model to training mode
			else:
				model.train(False)  # Set model to evaluate mode
	
			running_loss = 0.0
			running_corrects = 0
	
			# Iterate over data.
			for data in dset_loaders[phase]:
				# get the inputs
				inputs, labels = data
	
				# wrap them in Variable
				inputs, labels = Variable(inputs), Variable(labels)
	
				# zero the parameter gradients
				optimizer.zero_grad()
	
				# forward
				outputs = model(inputs)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
	
				# backward + optimize only if in training phase
				if phase == 'train':
					loss.backward()
					optimizer.step()
	
				# statistics
				running_loss += loss.data[0]
				running_corrects += torch.sum(preds == labels.data)
	
			epoch_loss = running_loss / dset_sizes[phase]
			epoch_acc = running_corrects / dset_sizes[phase]
	
			print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
	
		print()
	
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=2):
	lr = init_lr * (0.1**(epoch // lr_decay_epoch))
	
	if epoch % lr_decay_epoch == 0:
		print('LR is set to {}'.format(lr))
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	
	return optimizer

model = models.resnet18(pretrained=False)
model.load_state_dict(torch.load('result/resnet18-5c106cde.pth'))
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)
	
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
train_model(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
