
# */AIPND/aipnd-project/train.py
#                                                                             
# PROGRAMMER: Nguyen Thi Huong
# DATE CREATED: 19/09/2023


# PURPOSE: Train network based on pre-trained network and save checkpoint. Load
#          and preprocess (transform) training/validation/testing sets, map labels, load
#          pre-trained model (e.g. resnet18), create classifier and add to pre-trained 
#          model, train network (print training loss, validation loss, validation accuracy
#          during training to show progress) and finally save the trained model.
#
# Use argparse Expected Call with <> indicating expected user input:
#      python train.py 
#             data_dir <directory with images> 
#             --save_dir <directory for saving checkpoint.pth> 
#             --arch <model out of resnet list>
#             --learning rate <for training>
#             --dropout <for training>
#             --hidden_units <node size in classifier as list>
#             --epochs <for training>
#             --gpu <enable cuda>
# 
#   Example call:
#    python train.py flowers --arch resnet18 --learning_rate 0.001 --epochs 5 --dropout 0.2 --gpu
##

# import python modules

import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from torch.autograd import Variable

import argparse


def main():

    # Set up argument parser for console input
    parser = argparse.ArgumentParser(description='Udacity project')
    parser.add_argument('data_dir', help='directory containing sub-folders with data')
    parser.add_argument('--save_dir', help='directory for saving checkpoint', default='checkpoints')
    parser.add_argument('--arch', help='pre-trained model architecture', default='vgg16')
    parser.add_argument('--learning_rate', help='learning rate during learning', type=float, default=0.001)
    parser.add_argument('--dropout', help='dropout during learning', type=float, default=0.4)
    parser.add_argument('--hidden_units', help='List of number of nodes in hidden layers', nargs='+', type=int,
                          default=[256, 102])
    parser.add_argument('--epochs', help='Number of epochs for training', default=3, type=int)
    parser.add_argument('--gpu', help='Enable GPU', action='store_true')

    args = parser.parse_args()
    return args


    dataloaders['training'], dataloaders['testing'], dataloaders['validation'] = futils.load_data()

    futils.load_checkpoint(path)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    probabilities = futils.predict(path_image, model, number_of_outputs, power)

    labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])

    i = 0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
i += 1

print("!!! Task Completed Successfully !!!")