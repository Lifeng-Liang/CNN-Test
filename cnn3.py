import os
import time
import torch
import torch.nn as nn
from util import run, new_fc2, new_layer, weights_init

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.init = nn.BatchNorm2d(1)
		self.conv1 = new_layer(1, 16, 5, 1, 2, drop_rate=0.2)
		self.conv2 = new_layer(16, 32, 5, 1, 2, drop_rate=0.2)
		self.conv3 = new_layer(32, 64, 3, 1, 0, drop_rate=0.2)
		self.out = new_fc2(64*5*5, 1024, 512, 7)

	def forward(self, x):
		x = self.init(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		return self.out(x)


cnn = CNN()
cnn.apply(weights_init)
#optim = torch.optim.Adam(cnn.parameters(), lr=LR)
#optim =torch.optim.SGD(cnn.parameters(), lr=0.1, momentum=0.9)
optim = torch.optim.Adadelta(cnn.parameters())
run(cnn, optim, nn.CrossEntropyLoss(), 500, train_path='val', val_path='')
