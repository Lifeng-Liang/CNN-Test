import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)			# reproducible

# Hyper Parameters
EPOCH = 200						# train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 100
LR = 0.001						# learning rate

#	torchvision.transforms.RandomHorizontalFlip(),
#	torchvision.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))

trans = torchvision.transforms.Compose([
	torchvision.transforms.Grayscale(),
	torchvision.transforms.ToTensor()
	]
)

train_data = torchvision.datasets.ImageFolder('train', trans)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.ImageFolder('test', trans)
test_x = list(map(lambda x: x[0], test_data))
test_loader = Data.DataLoader(dataset=test_data, batch_size=4000, shuffle=True)
show_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=True)
test_y = list(map(lambda x: x[1], test_data.imgs))


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Sequential(				# input shape (1, 48, 48)
			nn.Conv2d(
				in_channels=1,					# input height
				out_channels=16,				# n_filters
				kernel_size=5,					# filter size
				stride=1,						# filter movement/step
				padding=2,						# if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
			),									# output shape (16, 48, 48)
			nn.ReLU(),							# activation
			nn.MaxPool2d(kernel_size=2),		# choose max value in 2x2 area, output shape (16, 24, 24)
		)
		self.conv2 = nn.Sequential(				# input shape (16, 24, 24)
			nn.Conv2d(16, 32, 5, 1, 2),			# output shape (32, 24, 42)
			nn.ReLU(),							# activation
			nn.MaxPool2d(2),					# output shape (32, 12, 12)
		)
		self.out = nn.Linear(32 * 12 * 12, 7)   # fully connected layer, output 10 classes

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = x.view(x.size(0), -1)				# flatten the output of conv2 to (batch_size, 32 * 12 * 12)
		output = self.out(x)
		return output, x						# return x for visualization


cnn = CNN()
print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()						# the target label is not one-hotted

# training and testing
for ts,(tx,ty) in enumerate(test_loader):
	for epoch in range(EPOCH):
		t0 = time.clock()
		for step, (x, y) in enumerate(train_loader):	# gives batch data, normalize x when iterate train_loader
			b_x = Variable(x)							# batch x
			b_y = Variable(y)							# batch y

			output = cnn(b_x)[0]						# cnn output
			loss = loss_func(output, b_y)				# cross entropy loss
			optimizer.zero_grad()						# clear gradients for this training step
			loss.backward()								# backpropagation, compute gradients
			optimizer.step()							# apply gradients

			if step % 100 == 0:
				test_output, last_layer = cnn(Variable(tx))
				pred_y = torch.max(test_output, 1)[1].data.squeeze()
				zy = torch.LongTensor(ty)
				accuracy = sum(pred_y == zy) / float(zy.size(0))
				print('Epoch: ', epoch, '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy, ' | et: ', time.clock() - t0)
				t0 = time.clock()
	break

# print 10 predictions from test data
for ts,(tx,ty) in enumerate(show_loader):
	test_output, _ = cnn(Variable(tx))
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	print(pred_y, 'prediction number')
	print(ty[:10].numpy(), 'real number')
	break
