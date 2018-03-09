# -*- coding: utf-8 -*-
# http://blog.csdn.net/zwx2445205419/article/details/79030001
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
import time

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(42),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize()
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(42),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize()
    ])
}

data_dir = r"."
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()


def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bn_x = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm2d(32, momentum=0.5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn_conv3 = nn.BatchNorm2d(64, momentum=0.5)
        self.fc1 = nn.Linear(in_features=5 * 5 * 64, out_features=2048)
        self.bn_fc1 = nn.BatchNorm1d(2048, momentum=0.5)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.bn_fc2 = nn.BatchNorm1d(1024, momentum=0.5)
        self.fc3 = nn.Linear(in_features=1024, out_features=7)

    def forward(self, x):
        x = self.bn_x(x)
        x = F.max_pool2d(F.tanh(self.bn_conv1(self.conv1(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.tanh(self.bn_conv2(self.conv2(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = F.max_pool2d(F.tanh(self.bn_conv3(self.conv3(x))), kernel_size=3, stride=2, ceil_mode=True)
        x = x.view(-1, self.num_flat_features(x))
        x = F.tanh(self.bn_fc1(self.fc1(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.tanh(self.bn_fc2(self.fc2(x)))
        x = F.dropout(x, training=self.training, p=0.4)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def test_model():
    inputs, labels = next(iter(dataloaders['train']))
    print(inputs.size())
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in classes])
    model = Model()
    if use_gpu:
        model = model.cuda()
    print(model)
    outputs = model(inputs)
    print(outputs)


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        torch.save(model, 'best_model.pkl')
        torch.save(model.state_dict(), 'model_params.pkl')


if __name__ == '__main__':
    # test_model()
    model = Model()
    print(model)
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, criterion, optimizer, num_epochs=500)