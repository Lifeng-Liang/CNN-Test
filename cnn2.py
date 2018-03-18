import os
import time
import torch
import torch.nn as nn
from util import Initializer

m = Initializer()

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		#self.init = nn.BatchNorm2d(1)
		self.conv1 = m.new_layer(1, 32, drop_rate=0.3)
		self.conv2 = m.new_layer(32, 64, drop_rate=0.3)
		self.conv3 = m.new_layer(64, 64, drop_rate=0.3)
		self.out = m.new_fc(64*6*6, 1024, 7)

	def forward(self, x):
		#x = self.init(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = x.view(x.size(0), -1)
		return self.out(x)

cnn = CNN()
#optim = torch.optim.Adam(cnn.parameters(), lr=LR)
optim = torch.optim.Adadelta(cnn.parameters())
m.run(cnn, optim, nn.CrossEntropyLoss(), 500)
