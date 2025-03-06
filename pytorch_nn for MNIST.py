import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pytorch_nn


# Define hyper parameters
input_size = 28 * 28
output_size = 10
learning_rate = 0.001
epochs = 100


# Load MNIST dataset
xy = np.loadtxt("mnist_data.csv", delimiter=",", dtype=np.float32, skiprows=1)

dataset = pytorch_nn.Data(xy)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)


# Initialize Neural Network
model = pytorch_nn.NN(input_size, output_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

""" for i, (images, labels) in enumerate(train_loader):
    print(images.shape, labels.shape)
    break """

# Train Neural Network
nn = model.fit(train_loader, loss_fn, optimizer, epochs)
print(nn[2])
