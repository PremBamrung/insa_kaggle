import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()

# For reproducibility
torch.manual_seed(SEED)
device = torch.device("cuda" if cuda else "cpu")

if cuda:
    torch.cuda.manual_seed(SEED)


# normalize will center around -1 1
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train = MNIST('./data', train=True, download=True, transform=trans, )
test = MNIST('./data', train=False, download=True, transform=trans, )


print(train)
# Create DataLoader
dataloader_args = dict(shuffle=True, batch_size=256, num_workers=4,
                       pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)


# Two CNN + two FC Layers NN
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # use nn.Sequential() for each layer1 and layer2, with nn.Conv2d + nn.ReLU + nn.MaxPool2d
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3), nn.ReLU(), nn.MaxPool2d(2))
        # use nn.Linear with 1000 neurons
        self.fc1 = nn.Linear(16 * 5 * 5, 1000)
        # use nn.Linear to output a one hot vector to encode the output
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # use reshape() to match the input of the FC layer1
        out = out.reshape(x.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        # use F.log_softmax() to normalize the output
        return F.log_softmax(out, dim=1)


model = Model()

model.to(device)
# use an optim.Adam() optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
losses = []

# Eval
evaluate_x = test_loader.dataset.data.type_as(torch.FloatTensor())
evaluate_y = test_loader.dataset.targets
evaluate_x = evaluate_x.unsqueeze_(1)

evaluate_x, evaluate_y = evaluate_x.to(device), evaluate_y.to(device)
train_size = len(train_loader.dataset)
batch_size = (train_size / 256) if (cuda) else (train_size / 64)

for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = F.cross_entropy(y_pred, target)
        losses.append(loss.cpu().item())
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Display
        if batch_idx % 100 == 1:
            print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch + 1,
                EPOCHS,
                batch_idx * len(data),
                train_size,
                100. * batch_idx / batch_size,
                loss.cpu().item()),
                end='')

    # display final evaluation for this epoch
    model.eval()
    output = model(evaluate_x)
    pred = output.data.max(1)[1]
    d = pred.eq(evaluate_y.data).cpu()
    accuracy = d.sum().item() / d.size()[0]

    print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Test Accuracy: {:.4f}%'.format(
        epoch + 1,
        EPOCHS,
        train_size,
        train_size,
        100. * batch_idx / batch_size,
        loss.cpu().item(),
        accuracy * 100,
        end=''))
