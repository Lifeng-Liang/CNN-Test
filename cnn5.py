import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from util import Initializer

m = Initializer(crop=True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = m.new_layer(1, 32, 3, 1, 2, drop_rate=0.5)
		self.conv2 = m.new_layer(32, 64)
		self.conv3 = m.new_layer(64, 128, 3, 1, 1, 3, 2, 1)
		self.conv4 = m.new_layer(128, 128)
		self.out = m.new_fc2(128*3*3, 1024, 512, 7)

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
