import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import torch.utils.data as dataloaders

ap = argparse.ArgumentParser(description='Udacity project> Train.py')
# Command Line arguments
ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest="dropout", action="store", default=0.4)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=3)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type=str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=256)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

import torch.utils
dataloaders = futils.load_data(where)

model, optimizer, criterion = torch.utils.nn_setup(structure, dropout, hidden_layer1, lr, power)

torch.utils.train_network(model, optimizer, criterion, epochs, 3, dataloaders['training'], power)

torch.utils.save_checkpoint(img_path, structure, hidden_layer1, dropout, lr)

print("!!! Training Sequence Completed !!!")
