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
EPOCH = 500						# train the training data n times, to save time, we just train 1 epoch
LR = 0.001						# learning rate
DOWNLOAD_MNIST = False
BATCH_SIZE = 64
TIME_STEP = 48
INPUT_SIZE = 48

trans = torchvision.transforms.Compose([
	torchvision.transforms.RandomHorizontalFlip(),
	torchvision.transforms.Grayscale(),
	torchvision.transforms.ToTensor()
	]
)

test_data = torchvision.datasets.ImageFolder('test', trans)
test_loader = Data.DataLoader(dataset=test_data, batch_size=2000, shuffle=True)
show_loader = Data.DataLoader(dataset=test_data, batch_size=10, shuffle=True)


class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		self.rnn = nn.LSTM(			# if use nn.RNN(), it hardly learns
			input_size=INPUT_SIZE,
			hidden_size=64,			# rnn hidden unit
			num_layers=2,			# number of rnn layer
			batch_first=True,		# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
		)

		self.out = nn.Linear(64, 7)

	def forward(self, x):
		# x shape (batch, time_step, input_size)
		# r_out shape (batch, time_step, output_size)
		# h_n shape (n_layers, batch, hidden_size)
		# h_c shape (n_layers, batch, hidden_size)
		r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

		# choose r_out at the last time step
		out = self.out(r_out[:, -1, :])
		return out

rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()						# the target label is not one-hotted

# training and testing
tx,ty = next(iter(test_loader))
print('Starting learning...')
t0 = time.time()
for epoch in range(EPOCH):
	train_data = torchvision.datasets.ImageFolder('train', trans)
	train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
	loss_data = 0.0
	count = 0
	for step, (x, y) in enumerate(train_loader):	# gives batch data, normalize x when iterate train_loader
		b_x = Variable(x.view(-1, 48, 48))			# batch x
		b_y = Variable(y)							# batch y
		output = rnn(b_x)							# cnn output
		loss = loss_func(output, b_y)				# cross entropy loss
		optimizer.zero_grad()						# clear gradients for this training step
		loss.backward()								# backpropagation, compute gradients
		optimizer.step()							# apply gradients
		loss_data += loss.data[0]
		count += 1

	loss_data /= count
	test_output = rnn(Variable(tx.view(-1, 48, 48)))
	pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
	zy = torch.LongTensor(ty)
	accuracy = sum(pred_y == zy) / float(zy.size(0))
	tend = time.time()
	print('Epoch: ', epoch, '| train loss: %.4f' % loss_data, '| test accuracy: %.2f' % accuracy, ' | et:', tend - t0)
	t0 = tend

# print 10 predictions from test data
tx,ty = next(iter(show_loader))
test_output = rnn(Variable(tx.view(-1, 48, 48)))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(ty.numpy(), 'real number')
