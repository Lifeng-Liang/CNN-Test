import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from util import run, new_fc2, new_layer, weights_init

def newLevel(inNum, outNum, k=3, s=1, p=1, pk=2, ps=2, pp=0):
	return nn.Sequential(
		nn.Conv2d(inNum, outNum, k, s, p),
		nn.MaxPool2d(pk, ps, pp),
		nn.ReLU(),
		nn.BatchNorm2d(outNum),
	)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.init = nn.Dropout2d()
		self.conv1 = newLevel(1, 16, 3, 1, 1)
		self.conv2 = newLevel(16, 32, 3, 1, 1)
		self.conv3 = newLevel(32, 64, 3, 1, 1)
		self.out = new_fc2(64*6*6, 1024, 512, 7)

	def forward(self, x):
		x = self.init(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		return self.out(x)


cnn = CNN()
#cnn.apply(weights_init)
#optim = torch.optim.Adam(cnn.parameters(), lr=LR)
optim = torch.optim.Adadelta(cnn.parameters())
#run(cnn, optim, nn.CrossEntropyLoss(), 500, train_path='val', val_path='')
run(cnn, optim, nn.CrossEntropyLoss(), 500)
