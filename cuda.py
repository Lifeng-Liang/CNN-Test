import os
import time
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as tf
from torch.nn.init import xavier_uniform as xavier

class Initializer():
	def __init__(self, batch_size=64, crop=False, train_path='train', test_path='test', val_path='val', num_workers=0):
		self.batch_size		= batch_size
		self.train_path		= train_path
		self.test_path		= test_path
		self.val_path		= val_path
		self.num_workers	= num_workers
		self.best_acc		= [0.0, 0.0, 0.0]
		self.save_files		= ['best.pkl', 'best_val.pkl', 'best_avg.pkl', 'snap.pkl']
		if(crop):
			self.train_trans = tf.Compose([tf.RandomResizedCrop(42), tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])
			self.test_trans = tf.Compose([tf.CenterCrop(42), tf.Grayscale(), tf.ToTensor()])
		else:
			self.train_trans = tf.Compose([tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])
			self.test_trans = tf.Compose([tf.Grayscale(), tf.ToTensor()])

	def run(self, cnn, optimizer, loss_func, EPOCH, min_acc = 50.0):
		print(cnn)

		train_loader = self.read(self.train_path, self.batch_size, self.train_trans)
		test_loader = self.read(self.test_path, self.batch_size, self.test_trans)
		if(self.val_path != ''):
			val_loader = self.read(self.val_path, self.batch_size, self.test_trans)

		tstart = time.time()
		val_acc = 0.0
		avg_acc = 0.0
		print('Start learning...')
		for epoch in range(EPOCH):
			loss_data = self.train(cnn, train_loader, optimizer, loss_func)
			test_acc = self.test(cnn, test_loader)
			if(self.val_path != ''):
				val_acc = self.test(cnn, val_loader)
				avg_acc = (test_acc + val_acc) / 2
			tend = time.time()
			tdura = tend - tstart
			print('Epoch:', epoch, '| loss: %.4f' % loss_data, '| test acc: %.2f' % test_acc, '| val acc: %.2f' % val_acc, '| avg acc: %.2f' % avg_acc, '| et: %.3f' % tdura)
			self.save(cnn, epoch, min_acc, [test_acc, val_acc, avg_acc])
			tstart = tend

		self.show(cnn)

	def save(self, cnn, epoch, min_acc, tas):
		for i,acc in enumerate(tas):
			if(acc > min_acc and acc > self.best_acc[i]):
				self.best_acc[i] = acc
				torch.save(cnn, self.save_files[i])
				print('Save %s for acc %.2f' % (self.save_files[i], acc))
		if(epoch > 0 and (epoch % 100) == 0):
			torch.save(cnn, self.save_files[3])
			print('Save %s for epoch %d' % (self.save_files[3], epoch))

	def read(self, folder, batch_size, trans):
		data = torchvision.datasets.ImageFolder(folder, trans)
		loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
		return loader

	def train(self, model, loader, optimizer, loss_func):
		model.train()
		loss_data = 0.0
		count = 0
		for _, (x, y) in enumerate(loader):
			b_x, b_y = Variable(x.cuda()), Variable(y.cuda())
			output = model(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()						# clear gradients for this training step
			loss.backward()								# backpropagation, compute gradients
			optimizer.step()							# apply gradients
			loss_data += loss.data.item()
			count += 1
		return loss_data / count

	def test(self, model, loader):
		model.eval()
		sumAcc = 0.0
		count = 0
		for _, (tx, ty) in enumerate(loader):
			zx, zy = Variable(tx.cuda()), torch.LongTensor(ty)
			test_output = model(zx)
			pred_y = self.get_pred(test_output)
			sumAcc += sum(pred_y == zy).cpu().numpy()
			count += zy.size(0)
		return sumAcc * 100.0 / count

	def show(self, model):
		model.eval()
		show_loader = self.read('test', 10, self.test_trans)
		sx, sy = next(iter(show_loader))
		test_output = model(Variable(sx))
		pred_y = self.get_pred(test_output)
		print(pred_y, 'prediction number')
		print(sy.numpy(), 'real number')

	def get_pred(self, output):
		return torch.max(output, 1)[1].data.cpu().numpy().squeeze()

	def fc(self, list, drop_rate=0.5):
		seq = nn.Sequential()
		size = len(list) - 1
		i = iter(range(100))
		for index in range(size-1):
			out_num = list[index+1]
			seq.add_module(str(next(i)), nn.Linear(list[index], out_num))
			seq.add_module(str(next(i)), nn.BatchNorm1d(out_num))
			seq.add_module(str(next(i)), nn.ReLU())
			if(drop_rate > 0.0):
				seq.add_module(str(next(i)), nn.Dropout(drop_rate))
		seq.add_module(str(next(i)), nn.Linear(list[size-1], list[size]))
		seq.add_module(str(next(i)), nn.LogSoftmax(dim=1))
		return seq

	def new_fc(self, in_num, num1, out_num, drop_rate=0.5):
		return nn.Sequential(
			nn.Linear(in_num, num1),
			nn.BatchNorm1d(num1),
			nn.ReLU(),
			nn.Dropout(drop_rate),
			nn.Linear(num1, out_num),
			nn.Softmax(dim=1)
		)

	def new_fc2(self, in_num, num1, num2, out_num, drop_rate=0.5):
		return nn.Sequential(
			nn.Linear(in_num, num1),
			nn.BatchNorm1d(num1),
			nn.ReLU(),
			nn.Dropout(drop_rate),
			nn.Linear(num1, num2),
			nn.BatchNorm1d(num2),
			nn.ReLU(),
			nn.Dropout(drop_rate),
			nn.Linear(num2, out_num),
			nn.Softmax(dim=1),
		)

	def new_layer(self, in_num, out_num, k=3, s=1, p=1, pk=2, ps=2, pp=0, drop_rate=0.0):
		if(drop_rate > 0.0):
			return nn.Sequential(
				nn.Conv2d(
					in_channels=in_num,
					out_channels=out_num,
					kernel_size=k,
					stride=s,
					padding=p,
				),
				nn.BatchNorm2d(out_num),
				nn.ReLU(),
				nn.Dropout2d(drop_rate),
				nn.MaxPool2d(kernel_size=pk, stride=ps, padding=pp),
			)
		else:
			return nn.Sequential(
				nn.Conv2d(in_num, out_num, k, s, p),
				nn.BatchNorm2d(out_num),
				nn.ReLU(),
				nn.MaxPool2d(kernel_size=pk, stride=ps, padding=pp),
			)

	def weights_init(self, m):
		classname=m.__class__.__name__
		if classname == 'Conv2d' or classname == 'Linear':
			print(classname, 'apply xavier')
			xavier(m.weight.data)
			#xavier(m.bias.data)
			m.bias.data.fill_(0.1)


m = Initializer(num_workers=0, crop=True)

class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = m.new_layer(1, 32, 3, 1, 0, 2, 2, 0, drop_rate=0.0)
		self.conv2 = m.new_layer(32, 64, drop_rate=0.0)
		self.conv3 = m.new_layer(64, 128, drop_rate=0.0)
		#self.conv4 = m.new_layer(128, 128)
		self.out = m.fc([128*5*5, 1024, 512, 7], drop_rate=0.4)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		#x = self.conv4(x)
		x = x.view(x.size(0), -1)
		return self.out(x)


cnn = CNN()
cnn = cnn.cuda()
#cnn.apply(m.weights_init)
optim = torch.optim.Adam(cnn.parameters())
#optim = torch.optim.Adadelta(cnn.parameters())
#run(cnn, optim, nn.CrossEntropyLoss(), 500, train_path='val', val_path='')
m.run(cnn, optim, nn.CrossEntropyLoss(), 5000)
