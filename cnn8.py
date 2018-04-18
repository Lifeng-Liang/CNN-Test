import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from util import Initializer

m = Initializer(num_workers=0, crop=True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = m.new_layer(1, 32, 3, 1, 0, 2, 2, 0, drop_rate=0.0)
		self.conv2 = m.new_layer(32, 64, drop_rate=0.0)
		self.conv3 = m.new_layer(64, 128, drop_rate=0.0)
		#self.conv4 = m.new_layer(128, 128)
		self.out = m.fc([128*5*5, 1024, 512, 7], drop_rate=0.0)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		#x = self.conv4(x)
		x = x.view(x.size(0), -1)
		return self.out(x)


cnn = CNN()
#cnn.apply(m.weights_init)
#optim = torch.optim.Adam(cnn.parameters(), lr=0.001)
#optim = torch.optim.Adadelta(cnn.parameters())
optim = torch.optim.SGD(cnn.parameters(), lr=0.01, momentum=0.9)
#run(cnn, optim, nn.CrossEntropyLoss(), 500, train_path='val', val_path='')
m.run(cnn, optim, nn.CrossEntropyLoss(), 5000)
