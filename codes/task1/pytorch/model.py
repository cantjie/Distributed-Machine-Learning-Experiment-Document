import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
from MyOptimizer import GdOptimizer, AdamOptimizer
import numpy as np
import argparse

class Net(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: (b, 1, 28, 28)
        """
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        # flatten the feature map
        out = out.flatten(1)
        # fc layer
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

def train(model, dataloader, optimizer, loss_fn, num_epochs=1, optim="sgd"):
    print("Start training ...")
    loss_total = 0.
    model.train()

    # code changed
    loss_records = []
    iteration_records = []

    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            # with dist_autograd.context() as context_id:
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()

            if i % 100 == 0:    
                print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))

            if optim=="sgd" and i % 20 == 19:    
                # print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                loss_records.append(loss_total / 20)
                iteration_records.append(i + 1)
                loss_total = 0.0

            elif optim=="adam":    
                # print('epoch: %d, iters: %5d, loss: %.3f' % (epoch + 1, i + 1, loss_total / 20))
                loss_records.append(loss_total)
                iteration_records.append(epoch * len(dataloader) + i + 1)
                loss_total = 0.0

            elif optim == "gd":
                loss_records.append(loss_total)
                iteration_records.append(epoch + 1)
                loss_total = 0.0
    
    # code changed
    np.save('./{}_loss_records'.format(optim), np.asanyarray(loss_records))
    np.save('./{}_itr_records'.format(optim), np.asanyarray(iteration_records))

    print("Training Finished!")

def test(model: nn.Module, test_loader):
    # test
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim", default="sgd", type=str, help="Set the optimizer", choices=['adam', 'gd', 'sgd'])

    args = parser.parse_args()

    return args



def main():
    model = Net(in_channels=1, num_classes=10)
    model.cuda()

    args = parse_args()

    DATA_PATH = "./data"

    transform = torchvision.transforms.Compose(
                [torchvision.transforms.ToTensor()]
                )

    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    # SGD setting
    train_loader_sgd = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)

    # ADAM setting
    train_loader_adam = torch.utils.data.DataLoader(train_set, batch_size=10000, shuffle=True)

    # GD setting
    train_loader_gd = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)

    MATCH_DICT = {
        "sgd": {
            "train_loader": train_loader_sgd,
            "epoch": 1,
            "optim": GdOptimizer(model.parameters(), lr=0.01)
        },
        "adam": {
            "train_loader": train_loader_adam,
            "epoch": 5,
            "optim": AdamOptimizer(model.parameters(), lr=0.01, b1=0.03, b2=0.05)
        },
        "gd": {
            "train_loader": train_loader_gd,
            "epoch": 25,
            "optim": GdOptimizer(model.parameters(), lr=0.01)
        },
    }

    train_loader = MATCH_DICT[args.optim]["train_loader"]

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    
    optimizer =  MATCH_DICT[args.optim]["optim"]

    train(model, train_loader, optimizer, loss_fn, optim=args.optim, num_epochs=MATCH_DICT[args.optim]["epoch"])
    test(model, test_loader)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()