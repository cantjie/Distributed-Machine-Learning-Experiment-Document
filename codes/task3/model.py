import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import argparse
import torch.distributed as dist 
import torch.multiprocessing as mp
import dist_utils
from sampler import MySampler

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


def train(model, dataloader, loss_fn, optimizer, num_epochs=2):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.
    model.train()

    dist_utils.init_parameters(model)

    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            inputs, labels = batch_data
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            # averge the gradients of model parameters
            dist_utils.average_gradients(model)
            optimizer.step()
            loss_total += loss.item()
            if i % 20 == 19:    
                print('Device: %d epoch: %d, iters: %5d, loss: %.3f' % (dist_utils.get_local_rank(), epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0

    print("Training Finished!")

def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--gpu', default="0", type=str, help='GPU ID')
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    DATA_PATH = "./data"
    # initialize process group
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    # construct the model
    model = Net(in_channels=1, num_classes=10)
    model.cuda()

    # construct the dataset
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

    sampler = MySampler(train_set, args.n_devices, args.rank, shuffle=True, seed=args.rank)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, sampler=sampler)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # construct the loss_fn and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train(model, train_loader, loss_fn, optimizer)
    test(model, test_loader)