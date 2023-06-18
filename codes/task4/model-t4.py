import os
from urllib import parse
import torch
from torch.distributed.distributed_c10d import get_rank, get_world_size 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

import argparse
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
from torch.distributed.rpc import rpc_sync
import dist_utils
import time
import pickle

class SubNetConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)

    def forward(self, x_rref):
        """
        Write your code here!
        """
        out = F.max_pool2d(F.relu(self.conv1(x_rref)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        return out
        pass

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class SubNetFC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x_rref):
        """
        Write your code here!
        """
        out = x_rref.flatten(1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        pass

    def parameter_rrefs(self):
        return [rpc.RRef(p) for p in self.parameters()]


class ParallelNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        # 分别远程声明SubNetConv和SubNetFC
        """
        Write your code here!
        """
        self.conv_net = rpc.remote("worker1", SubNetConv, args=(in_channels,))
        self.fc_net = rpc.remote("worker2", SubNetFC, args=(num_classes,))
        pass

    def forward(self, x):
        """
        Write your code here!
        """
        out = self.conv_net.rpc_sync().forward(x)
        out = self.fc_net.rpc_sync().forward(out)
        return out
        pass

    def parameter_rrefs(self):
        """
        Write your code here!
        """
        return self.conv_net.rpc_sync().parameter_rrefs() + self.fc_net.rpc_sync().parameter_rrefs()
        pass

def train(model, dataloader, loss_fn, optimizer, num_epochs=2):
    print("Device {} starts training ...".format(dist_utils.get_local_rank()))
    loss_total = 0.
    model.train()
    dist_utils.init_parameters(model)
    loss_path = []
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(dataloader):
            """
            Write your code here!
            """
            inputs, labels = batch_data
            with dist_autograd.context() as context_id:
                model.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                dist_autograd.backward(context_id,[loss])
                optimizer.step(context_id)
            loss_total += loss.item()
            loss_path.append(loss.item())
            if i % 20 == 19:  
                print('Device: %d epoch: %d, iters: %5d, loss: %.3f' % (dist_utils.get_local_rank(), epoch + 1, i + 1, loss_total / 20))
                loss_total = 0.0
            pass
    
    print("Training Finished!")
    return loss_path


def test(model: nn.Module, test_loader):
    model.eval()
    size = len(test_loader.dataset)
    correct = 0
    print("testing ...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, size,
        100 * correct / size))


def main():
    args = parse_args()
    dist_utils.dist_init(args.n_devices, args.rank, args.master_addr, args.master_port)
    DATA_PATH = "./data"
    if args.rank == 0:
        
        rpc.init_rpc("worker0", rank=args.rank, world_size=args.n_devices)
        # construct the model
        model = ParallelNet(in_channels=1, num_classes=10)
        # construct the dataset
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )
        train_set = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

        # construct the loss_fn and optimizer
        loss_fn = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        dist_optimizer = DistributedOptimizer(torch.optim.SGD, model.parameter_rrefs(), lr=0.01)
    
        start_time = time.time()
        loss = train(model, train_loader, loss_fn, dist_optimizer)
        end_time = time.time()
        with open('loss.pkl', 'wb') as file:
            pickle.dump(loss, file)
        print("时间花费{:.2f}".format(end_time-start_time))
        test(model, test_loader)
    
    elif args.rank == 1:
        rpc.init_rpc("worker1", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker1...")

    elif args.rank == 2:
        rpc.init_rpc("worker2", rank=args.rank, world_size=args.n_devices)
        print("Training on the worker2...")

    rpc.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_devices", default=1, type=int, help="The distributd world size.")
    parser.add_argument("--rank", default=0, type=int, help="The local rank of device.")
    parser.add_argument('--master_addr', default='localhost', type=str,help='ip of rank 0')
    parser.add_argument('--master_port', default='12355', type=str,help='ip of rank 0')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()