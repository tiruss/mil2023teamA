#!/usr/bin/env python
# coding: utf-8
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import argparse
from models import *
from utils import timeit


# %%
random_seed = 1234
torch.manual_seed(random_seed)

#class_labels = ['happy', 'suprise', 'angry','anxious', 'hurt', 'sad', 'neutral']
class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
                     '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}


def get_dataloader(args):
    classes_train = os.listdir(args.data_path + "/test")
    classes_valid = os.listdir(args.data_path + "/test")
    print("Data Folders -", os.listdir(args.data_path))
    print("Train Folders -", classes_train)
    print("Validation Folders -", classes_valid)
 
    image_size = (args.image_size, args.image_size)

    # Data transforms (Gray Scaling & data augmentation)
    train_transforms = tt.Compose([
        tt.Resize(image_size),
        tt.Grayscale(num_output_channels=args.image_channel),
        #tt.RandomCrop(48, padding=4, padding_mode='reflect'),
        tt.RandomHorizontalFlip(),
        tt.RandomRotation(6),
        tt.ColorJitter(),
        tt.ToTensor(),
        #tt.Normalize((0.5), (0.5), inplace=True)
    ])

    valid_transforms = tt.Compose([
        tt.Grayscale(num_output_channels=args.image_channel),
        tt.Resize(image_size),
        tt.ToTensor(),
        #tt.Normalize((0.5), (0.5), inplace=True)
    ])

    # Datasets from ImageFolder
    train_dataset = ImageFolder(args.data_path + '/test', train_transforms)
    valid_dataset = ImageFolder(args.data_path + '/test', valid_transforms)

    print("Classes -", train_dataset.classes)
    print("Classes (dict) -", train_dataset.class_to_idx)

    print("Number of Train Images -", len(train_dataset))
    print("Number of Validation Images -", len(valid_dataset))

    # PyTorch DataLoader
    train_dataloader = DataLoader(train_dataset, args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size,
                                  num_workers=args.num_workers, pin_memory=True, shuffle=False)

    return train_dataloader, valid_dataloader


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break
 
def get_device(args):
    if args.gpu and torch.cuda.is_available():
        #return torch.device('cuda')
        return torch.device('cuda:' + args.cuda_idx)
    else:
        return torch.device('cpu')


def train_epoch(model, dataset_loader, epoch, device, optimizer, criterion):
    running_loss = 0.0

    model.train()

    for i, (data, target) in enumerate(dataset_loader):
        inputs, labels = data, target
        inputs = inputs.float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)

        # we may use nll_loss or cross_entropy
        #loss = F.nll_loss(outputs, labels)
        loss = F.cross_entropy(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss


def evaluation(model, dataset_loader, device, criterion):
    correct = 0
    valid_loss = 0

    model.eval()

    with torch.no_grad():
        for i, (data, target) in enumerate(dataset_loader):

            images, labels = data, target
            images = images.float()
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            #valid_loss += F.nll_loss(outputs, labels).item()
            valid_loss += F.cross_entropy(outputs, labels).item()

            pred = outputs.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    accuracy = 100 * correct / len(dataset_loader.dataset)
    return valid_loss, accuracy


def load_model(args, learning_rate, device):
    checkpoint = torch.load(args.model_path)

    model = getModel(args.model)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    optimizer = getOptimizer(args.optimizer, model)
    optimizer.load_state_dict(checkpoint['optimizer'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return model, optimizer

def restart_training(best_model, learning_rate, device):
    model, optimizer = load_model(best_model, learning_rate, device)
    return model, optimizer


def getModel(modelName):    
    modelName = modelName.lower()
    if modelName == "vgg19":
        print("Model - VGG19")
        return VGG("VGG19")
    elif modelName == "vgg22":
        print("Model - VGG22")
        return VGG("VGG22")
    elif modelName == "vgg24":
        print("Model - VGG24")
        return VGG("VGG24")
    elif modelName == "resnet18":
        print("Model - ResNet18")
        return ResNet18()
    elif modelName == "emotionnet":
        print("Model - EmotionNet")
        return EmotionNet()
    elif modelName == "resemotionnet":
        print("Model - ResEmotionNet")
        return ResEmotionNet()
    elif modelName == "efficientnet-b4":
        print("Model - EfficientNet-b4")
        return EfficientNet.from_name('efficientnet-b4')
    elif modelName == "efficientnet-b5":
        print("Model - EfficientNet-b5")
        return EfficientNet.from_name('efficientnet-b5')

    print("Invalid model input:", modelName)
    print("Use instead: CNN")
    return CNN()


def getOptimizer(args, model):
    optName = args.optimizer.lower()
    if optName == "sgd":
        print("Optimizer - SGD")
        return optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif optName == "adam":
        print("Optimizer - Adam")
        return optim.Adam(model.parameters(), lr=0.001)

    # by default
    print("Optimizer - Adadelta")
    return optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, eps=1e-8)


@timeit
def train(args):

    device = get_device(args)
    print("Device -", device)

    train_dataloader, valid_dataloader = get_dataloader(args)

    model = getModel(args.model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = getOptimizer(args, model)

    # when we need lr scheduler
    #scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.01,  max_lr=0.1)                             
    
    loss_es = []
    best_accuracy = 0.0
    best_model = -1

    epoch = 1
    print("Epoch\tTrain Loss\tValidation Loss\tValidation Acc\tBest Model")
    while epoch <= args.epochs:
        running_loss = train_epoch(model, train_dataloader, epoch,
                             device, optimizer, criterion)
        valid_loss, accuracy = evaluation(model, valid_dataloader, device, criterion)
        
        #scheduler.step()

        loss_es.append((running_loss, valid_loss, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = epoch

            # write the best model as a file
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, args.model_path)

        print('{}\t{:.5f}\t{:.5f}\t{:.3f}\t\t{}'.format(
            epoch,
            running_loss,
            valid_loss,
            accuracy,
            best_model
        ))
        epoch += 1

    print('Trainig Complete.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action='store',
                        default='data', help='path of data folder')
    parser.add_argument('--model_path', action='store',
                        default='model_eff.pth', help='path of model file')
    parser.add_argument('--model', action='store',
                        default='emotionnet', help='network architecture')
    parser.add_argument('--optimizer', action='store',
                        default='adam', help='optimizer')
    
    # cnn, resnet, vgg19, vgg22: 48 | vgg24: 96 | efficientnet: 224, any
    parser.add_argument('--image_size', action='store', type=int,
                        default=48, help='input image size of the network')
    
    # 3 for efficientnet, 1 for the rest
    parser.add_argument('--image_channel', action='store', type=int,
                        default=1, help='input image layers')

    parser.add_argument('--gpu', action='store_true',
                        default=True, help='set a switch to use GPU')
    parser.add_argument('--cuda_idx', action='store',
                        default="0", help='set GPU index for multi-GPU')                 
    parser.add_argument('--epochs', action='store', type=int,
                        default=50, help='number of epochs')
    parser.add_argument('--lr', action='store', type=float,
                        default=0.1, help='learning rate')
    parser.add_argument('--batch_size', action='store', type=int,
                        default=64, help='size of batch')
    parser.add_argument('--num_workers', action='store', type=int,
                        default=4, help='number of workers')
    #parser.add_argument('--max_lr', action='store', type=float,
    #                    default=0.1, help='maximum learning rate')
    #parser.add_argument('--grad_clip', action='store', type=float, default=0.2,
    #                    help='clips gradient of an iterable of parameters at specified value')
    args = parser.parse_args()
    print("Input Arguments -", args)
 
    train(args)


# %%
if __name__ == "__main__":
    main()
