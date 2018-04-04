from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.onnx
import torchvision
import torchvision.transforms as tf
import torch.utils.data as Data

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



def read(folder, batch_size, trans):
    data = torchvision.datasets.ImageFolder(folder, trans)
    loader = Data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=0)
    return loader

def get_pred(output):
    return torch.max(output, 1)[1].data.numpy().squeeze()


test_trans = tf.Compose([tf.CenterCrop(42), tf.RandomHorizontalFlip(), tf.Grayscale(), tf.ToTensor()])
test_loader = read('test', 1, test_trans)

model = torch.load('best.pkl')
model.eval()
sx, sy = next(iter(test_loader))
test_output = model(Variable(sx))
pred_y = get_pred(test_output)
print('-----', test_output, test_output.data.numpy())
print('-----', torch.max(test_output, 1)[1].data.numpy())
print(pred_y, 'prediction number')
print(sy.numpy(), 'real number')

torch.onnx.export(model, Variable(sx), "best.onnx", verbose=True)
