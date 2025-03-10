# Classes and Functions for Neural Networks in PyTorch
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Create Neural Network Class
class NN(nn.Module):
    def __init__(self, input_size, output_size, activation_fn=nn.ReLU()):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.activation_fn = activation_fn
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            self.activation_fn,
            nn.Linear(512, 512),
            self.activation_fn,
            nn.Linear(512, output_size),
            self.activation_fn,
        )

    # Forward pass in NN layers
    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits

    # Train NN
    def fit(self, train_loader, optimizer, epochs):
        start = time.time()
        loss_list = []

        if self.activation_fn == nn.Sigmoid:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

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
            f"Finished Training with {self.activation_fn} in {end - start:.2f} seconds for {epochs} epochs and learning rate of {optimizer.param_groups[0]['lr']}"
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


# Calculate accuracy
def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():  # deactivate gradient calculation for inference
        for images, labels in data_loader:
            images = images.float()
            labels = labels.squeeze(dim=1).long()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total  # accuracy in percentage
