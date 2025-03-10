import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import pytorch_nn


# Define hyper parameters
input_size = 28 * 28
output_size = 10

epochs = 100
learning_rates = [0.01, 0.001, 0.0001, 0.00001]
activation_functions = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "LeakyReLu-0.01": nn.LeakyReLU(negative_slope=0.01),
    "LeakyReLu-0.05": nn.LeakyReLU(negative_slope=0.05),
    "LeakyReLu-0.1": nn.LeakyReLU(negative_slope=0.1),
    "LeakyReLu-0.5": nn.LeakyReLU(negative_slope=0.5),
    "PreLU": nn.PReLU(),
    "ELU-0.1": nn.ELU(alpha=0.1),
    "ELU-0.02": nn.ELU(alpha=0.2),
    "ELU-0.03": nn.ELU(alpha=0.3),
}


# Load MNIST dataset
xy = np.loadtxt(
    "./pytorch_approach/mnist_data.csv", delimiter=",", dtype=np.float32, skiprows=1
)

dataset = pytorch_nn.Data(xy)

# Split dataset into train and test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

""" # 1. Simple Tests
## Initialize Neural Network
model = pytorch_nn.NN(input_size, output_size, activation_fn=nn.ReLU())
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rates[0])

## Train Neural Network
nn = model.fit(train_loader, optimizer, epochs)

df_loss = pd.DataFrame(nn[2])
df_loss.to_csv("loss.csv", index=False)
print(df_loss)

## Performance
train_acc = pytorch_nn.calculate_accuracy(model, train_loader)
test_acc = pytorch_nn.calculate_accuracy(model, test_loader)

print(f"Train Accuracy: {train_acc:.2f}%")
print(f"Test Accuracy: {test_acc:.2f}%") """

# 2. Tests with different Activation Functions and Learning rates
df_performance = pd.DataFrame(
    columns=[
        "Learning Rate",
        "Activation Function",
        "Loss",
        "Duration",
        "Train Accuracy",
        "Test Accuracy",
    ]
)

for lrate in learning_rates:
    for key, elements in activation_functions.items():
        results = [lrate, key]

        # initialize model
        model = pytorch_nn.NN(input_size, output_size, activation_fn=elements)

        optimizer = optim.SGD(model.parameters(), lr=lrate)

        # Train Neural Network
        nn = model.fit(train_loader, optimizer, epochs)

        train_acc = pytorch_nn.calculate_accuracy(model, train_loader)
        test_acc = pytorch_nn.calculate_accuracy(model, test_loader)

        # Save results
        df_loss = pd.DataFrame(nn[2])
        df_loss.to_csv(
            f"./pytorch_approach/loss_files/{datetime.datetime.today().strftime('%Y-%m-%d %H-%M')}_loss_{key}_{lrate}_{epochs}.csv",
            index=False,
        )

        results.append(nn[0])
        results.append(nn[1])
        results.append(train_acc)
        results.append(test_acc)

        df_performance.loc[len(df_performance)] = results


print(df_performance.head(10))
df_performance.to_csv(
    f"./pytorch_approach/output/{datetime.datetime.today().strftime('%Y-%m-%d %H-%M')}_performance.csv",
    index=False,
)

# Save the model
torch.save(model.state_dict(), "./pytorch_approach/output/model.pth")
