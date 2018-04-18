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


cnn = torch.load('best.pkl')
loader = m.read('final/zhou', 64, tf.Compose([tf.Resize((42,42)), tf.Grayscale(), tf.ToTensor()]))

cnn.eval()
sx, sy = next(iter(loader))
test_output = cnn(Variable(sx))
pred_y = m.get_pred(test_output)
for x in test_output.data.numpy():
	for y in x:
		print('%.2f' % ((10**y)*100), end=', ')
	print('')
print(pred_y, 'prediction number')
print(sy.numpy(), 'real number')
