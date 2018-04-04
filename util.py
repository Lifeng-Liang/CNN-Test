import time
import torch
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
		self.best_acc		= 0.0
		self.test_acc		= 0.0
		if(crop):
			self.train_trans = tf.Compose([tf.RandomResizedCrop(42), tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])
			self.test_trans = tf.Compose([tf.CenterCrop(42), tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])
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
		print('Start learning...')
		for epoch in range(EPOCH):
			loss_data = self.train(cnn, train_loader, optimizer, loss_func)
			test_acc = self.test(cnn, test_loader)
			if(self.val_path != ''):
				val_acc = self.test(cnn, val_loader)
			tend = time.time()
			tdura = tend - tstart
			best_acc = (test_acc + val_acc) / 2
			print('Epoch:', epoch, '| loss: %.4f' % loss_data, '| test acc: %.2f' % test_acc, '| val acc: %.2f' % val_acc, '| avg acc: %.2f' % best_acc, '| et: %.3f' % tdura)
			if(test_acc > min_acc and test_acc > self.test_acc):
				self.test_acc = test_acc
				torch.save(cnn, 'best.pkl')
				print('Save best.pkl for test acc %.2f' % test_acc)
			if(val_acc > 0.0 and best_acc > min_acc and best_acc > self.best_acc):
				self.best_acc = best_acc
				torch.save(cnn, 'best_avg.pkl')
				print('Save best_avg.pkl for avg acc %.2f' % best_acc)
			tstart = tend

		self.show(cnn)

	def read(self, folder, batch_size, trans):
		data = torchvision.datasets.ImageFolder(folder, trans)
		loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
		return loader

	def train(self, model, loader, optimizer, loss_func):
		model.train()
		loss_data = 0.0
		count = 0
		for _, (x, y) in enumerate(loader):
			b_x, b_y = Variable(x), Variable(y)
			output = model(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()						# clear gradients for this training step
			loss.backward()								# backpropagation, compute gradients
			optimizer.step()							# apply gradients
			loss_data += loss.data[0]
			count += 1
		return loss_data / count

	def test(self, model, loader):
		model.eval()
		sumAcc = 0
		count = 0
		for _, (tx, ty) in enumerate(loader):
			zx, zy = Variable(tx), torch.LongTensor(ty)
			test_output = model(zx)
			pred_y = self.get_pred(test_output)
			sumAcc += sum(pred_y == zy)
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
		return torch.max(output, 1)[1].data.numpy().squeeze()

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
