# Classes and Functions for Neural Networks in PyTorch
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd


# Create Neural Network Class
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

    def fit(self, train_loader, loss_fn, optimizer, epochs):
        start = time.time()
        loss_list = []

        for epoch in range(epochs):
            dict_cc = {}
            for i, (images, labels) in enumerate(train_loader):
                labels = labels.type(torch.LongTensor)

                outputs = self(images)
                loss = loss_fn(outputs, labels.squeeze(dim=1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
            dict_cc["Epochs"] = epoch + 1
            dict_cc["Loss"] = round(loss.item(), 4)
            loss_list.append(dict_cc)
        end = time.time()

        print(
            f"Finished Training in {end - start:.2f} seconds for {epochs} epochs and learning rate of {optimizer.param_groups[0]['lr']}"
        )

        return round(loss.item(), 4), round(end - start, 2), loss_list


# Create Dataset Class
class Data(Dataset):
    def __init__(self, xy):
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
