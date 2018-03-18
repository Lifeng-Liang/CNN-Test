import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from util import Initializer

m = Initializer()

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
		self.conv1 = m.new_layer(1, 32)
		self.conv2 = m.new_layer(32, 64)
		self.conv3 = m.new_layer(64, 128)
		self.conv4 = m.new_layer(128, 256)
		self.out = m.new_fc2(256*3*3, 1024, 512, 7)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(0), -1)
		return self.out(x)


cnn = CNN()
#cnn.apply(weights_init)
#optim = torch.optim.Adam(cnn.parameters(), lr=LR)
optim = torch.optim.Adadelta(cnn.parameters())
#run(cnn, optim, nn.CrossEntropyLoss(), 500, train_path='val', val_path='')
m.run(cnn, optim, nn.CrossEntropyLoss(), 500)
