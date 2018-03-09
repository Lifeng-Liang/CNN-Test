import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as tf
from torch.nn.init import xavier_uniform as xavier

BATCH_SIZE = 64

def run(cnn, optimizer, loss_func, EPOCH, train_path='train', test_path='test', val_path='val'):
	print(cnn)

	train_loader = read(train_path)
	test_loader = read(test_path)
	if(val_path != ''):
		val_loader = read(val_path)

	tstart = time.time()
	val_acc = 0.0
	print('Start learning...')
	for epoch in range(EPOCH):
		loss_data = train(cnn, train_loader, optimizer, loss_func)
		test_acc = test(cnn, test_loader)
		if(val_path != ''):
			val_acc = test(cnn, val_loader)
		tend = time.time()
		tdura = tend - tstart
		print('Epoch:', epoch, '| loss: %.4f' % loss_data, '| test acc: %.2f' % test_acc, '| val acc: %.2f' % val_acc, '| et: %.3f' % tdura)
		tstart = tend

	show(cnn)

def read(folder, batchSize=BATCH_SIZE, trans = tf.Compose([tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])):
	data = torchvision.datasets.ImageFolder(folder, trans)
	loader = Data.DataLoader(dataset=data, batch_size=batchSize, shuffle=True)
	return loader

def train(model, loader, optimizer, loss_func):
	model.train()
	loss_data = 0.0
	count = 0
	for _, (x, y) in enumerate(loader):
		b_x, b_y = Variable(x), Variable(y)
		output = model(b_x)
		loss = loss_func(output, b_y)
		optimizer.zero_grad()						# clear gradients for this training step
		loss.backward()							# backpropagation, compute gradients
		optimizer.step()							# apply gradients
		loss_data += loss.data[0]
		count += 1
	return loss_data / count

def test(model, loader):
	model.eval()
	sumAcc = 0.0
	count = 0
	for _, (tx, ty) in enumerate(loader):
		zx, zy = Variable(tx), torch.LongTensor(ty)
		test_output = model(zx)
		pred_y = get_pred(test_output)
		a = sum(pred_y == zy) / float(zy.size(0))
		sumAcc += a
		count += 1
	return sumAcc * 100 / count

def show(model):
	model.eval()
	show_loader = read('test', 10)
	sx, sy = next(iter(show_loader))
	test_output = model(Variable(sx))
	pred_y = get_pred(test_output)
	print(pred_y, 'prediction number')
	print(sy.numpy(), 'real number')

def get_pred(output):
    return torch.max(output, 1)[1].data.numpy().squeeze()

def new_fc(in_num, num1, out_num):
	return nn.Sequential(
		nn.Linear(in_num, num1),
		nn.BatchNorm1d(num1),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(num1, out_num),
		nn.Softmax(dim=1),
	)

def new_fc2(in_num, num1, num2, out_num):
	return nn.Sequential(
		nn.Linear(in_num, num1),
		nn.BatchNorm1d(num1),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(num1, num2),
		nn.BatchNorm1d(num2),
		nn.ReLU(),
		nn.Dropout(),
		nn.Linear(num2, out_num),
		nn.Softmax(dim=1),
	)

def new_layer(in_num, out_num, k=3, s=1, p=1, pk=2, ps=2, pp=0, drop_rate=0.0):
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

def weights_init(m):
	classname=m.__class__.__name__
	if classname.find('Conv') != -1:
		print(classname)
		xavier(m.weight.data)
		#xavier(m.bias.data)
		m.bias.data.fill_(0.1)
