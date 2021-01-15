from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import collections
import time
import argparse
import sys
import pickle
import numpy as np
import collections
import time
from models.resnet_cifar import *
from models.wrn import Network
from models.mobilenetv2 import MobileNetV2
from models.densenet import DenseNet121
from models.cnn import Net

parser = argparse.ArgumentParser(description='PyTorch Grafting Training')
# basic setting
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--r', default=None, type=int)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--s', default='1', type=str)
parser.add_argument('--arch', default='ResNet56', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--print_frequence', default=100, type=int)
# Grafting setting
parser.add_argument('--a', default=0.4, type=float)
parser.add_argument('--c', default=10, type=int)
parser.add_argument('--num', default=1, type=int)
parser.add_argument('--i', default=1, type=int)
# Distillation setting
parser.add_argument('--teacher_arch', nargs='+', type=str, )
parser.add_argument('--teacher_dir', nargs='+', type=str, )
parser.add_argument('--T', default=2, type=float)
# Increase models diversity
parser.add_argument('--cos', action="store_true", default=False)
args = parser.parse_args()
print(args)
print('Session:%s\tModel:%d\tPID:%d' % (args.s, args.i, os.getpid()))
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
def creat_model(arch):
    if arch=='resnet20':
        return resnet20(args.num_classes).to(args.device)
    elif arch=='resnet32':
        return resnet32(args.num_classes).to(args.device)
    elif arch=='resnet56':
        return resnet56(args.num_classes).to(args.device)
    elif arch=='resnet110':
        return resnet110(args.num_classes).to(args.device)
    elif arch=='mobilenetv2':
        return MobileNetV2(args.num_classes).to(args.device)
    elif arch=='densenet':
        return DenseNet121(args.num_classes).to(args.device)
    elif arch=='wrn':
        return Network(args.num_classes).to(args.device)
    elif arch=='cnn':
        return Net(args.num_classes).to(args.device)

model = creat_model(args.arch)
print('Initialization  model completed!')
multi_teacher=[]
if args.teacher_arch and args.teacher_dir:
    for arch,dir in zip(args.teacher_arch,args.teacher_dir):
        teacher = creat_model(arch).eval()
        teacher.load_state_dict(torch.load(dir))
        print('Initialization teacher %s with %s completed!'%(arch,dir))
        multi_teacher.append(teacher)
start_epoch = 0
best_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# Data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.num_classes == 10:
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.num_classes == 100:
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
if args.cos == True:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * trainloader.__len__())
else:
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
def entropy(x, n=10):
    x = x.reshape(-1).cpu()
    scale = (x.max() - x.min()) / n
    entropy = 0
    for i in range(n):
        p = torch.sum((x >= x.min() + i * scale) * (x < x.min() + (i + 1) * scale), dtype=torch.float) / len(x)
        if p != 0:
            entropy -= p * torch.log(p)
    return float(entropy)


def grafting(model, epoch):
    torch.save(model.state_dict(), '%s/ckpt%d_%d.t7' % (args.s, args.i % args.num, epoch))
    while True:
        try:
            checkpoint = torch.load('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch))
            os.remove('%s/ckpt%d_%d.t7' % (args.s, args.i - 1, epoch))
            break
        except:
            time.sleep(10)
    odict = collections.OrderedDict()
    for i, (key, u) in enumerate(checkpoint.items()):
        if 'conv' in key:
            w = round(args.a / np.pi * np.arctan(args.c * (entropy(u) - entropy(checkpoint[key]))) + 0.5, 2)
        odict[key] = u * w + checkpoint[key] * (1 - w)
    model.load_state_dict(odict)


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '%s/best_%d.t7' % (args.s, args.i))
    print('Network:%d    epoch:%d    accuracy:%.3f    best:%.3f' % (args.i, epoch, acc, best_acc))


def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #################################Distillation###############################
        if args.teacher_arch and args.teacher_dir:
            for teacher in multi_teacher:
                teachers_outputs=teacher(inputs)
                loss +=(- F.log_softmax(outputs / args.T, 1) * F.softmax(teachers_outputs / args.T, 1)).sum(dim=1).mean() * args.T * args.T
        ############################################################################
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % args.print_frequence == args.print_frequence - 1 or args.print_frequence == trainloader.__len__() - 1:
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        if args.cos == True:
            lr_scheduler.step()
    if args.cos == False:
        lr_scheduler.step()
        


if __name__ == '__main__':
    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        ####################Grafting################################################
        if args.num > 1:
            grafting(model, epoch)
        ############################################################################