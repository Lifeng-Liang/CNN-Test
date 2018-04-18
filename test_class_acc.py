import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torchvision.transforms as tf
from util import Initializer

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


cnn = torch.load('best_val.pkl')
loader = m.read('val', 64, tf.Compose([tf.CenterCrop(42), tf.Grayscale(), tf.ToTensor()]))

accs = [[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
		[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]]
def sumit(py, ty):
	for i in range(len(py)):
		x = py[i]
		y = ty[i]
		accs[y][x][0] += 1
		for n in range(7):
			accs[y][n][1] += 1
		
def same(ty, n):
	r = []
	for x in ty:
		if(x == n):
			r.append(x)
		else:
			r.append(-1)
	return r

accs2 = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
def sumit2(py, ty, n):
	tty = same(ty, n)
	count = sum(ty == n)
	acc = sum(py == tty)
	accs2[n][1] += count
	accs2[n][0] += acc

cnn.eval()
sumAcc = 0
count = 0
for _, (tx, ty) in enumerate(loader):
	zx, zy = Variable(tx), torch.LongTensor(ty)
	test_output = cnn(zx)
	pred_y = m.get_pred(test_output)
	for i in range(7):
		sumit2(pred_y, ty, i)
	sumit(pred_y, ty)
	sumAcc += sum(pred_y == ty)
	count += zy.size(0)
print(sumAcc * 100.0 / count)
print(accs2)
for i in accs2:
	print('%.2f'%(i[0]*100.0/i[1]),end=' ')
print('')
print('----------------')
for line in accs:
	print(line)
print('----------------')
for line in accs:
	for item in line:
		print('%4.2f'%(item[0]*100.0/item[1]),end=' ')
	print('')
print('----------------')
need_sort = [
	['生气',accs[0][0],0.0],
	['厌恶',accs[1][1],0.0],
	['害怕',accs[2][2],0.0],
	['高兴',accs[3][3],0.0],
	['悲伤',accs[4][4],0.0],
	['惊讶',accs[5][5],0.0],
	['中性',accs[6][6],0.0]]
for p in need_sort:
	p[2] = p[1][0]/p[1][1]
def comp(x, y):
	if x[2] < y[2]:
		return 1
	elif x[2] > y[2]:
		return -1
	else:
		return 0
std = sorted(need_sort, key=lambda s:s[2], reverse=True)
print(std)
print('----------------')
for p in std:
	print('%s（%.2f%%，%d/%d），' %(p[0], p[2]*100, p[1][0], p[1][1]), end='')
print('')