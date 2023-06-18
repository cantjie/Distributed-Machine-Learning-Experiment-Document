import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler
import random
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torchvision import transforms
from torchvision import datasets
import os
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#下载数据集并提取
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=False, transform=transform)  
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=False, transform=transform)  # train=True训练集，=False测试集

#构建CNN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）



class DistributedRandomSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def __iter__(self):
        # 对数据索引进行随机采样
        indices = list(range(self.num_samples))
        random.shuffle(indices)

        # 计算每个GPU应该处理的数据索引范围
        indices = indices[self.rank:self.num_samples:self.world_size]

        return iter(indices)

    def __len__(self):
        return len(self.dataset) // self.world_size
    
    def set_epoch(self, epoch):
        # 在每个epoch之前设置新的随机种子
        random.seed(epoch)

class DistributedRandomSplitSampler(Sampler):
    def __init__(self, dataset, train_ratio=0.8):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = len(dataset)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.train_ratio = train_ratio

    def __iter__(self):
        # 对数据索引进行随机划分
        indices = list(range(self.num_samples))
        random.shuffle(indices)

        # 计算每个GPU应该处理的数据索引范围
        train_size = int(self.num_samples * self.train_ratio)
        train_indices = indices[self.rank:train_size:self.world_size]
        val_indices = indices[train_size:self.num_samples:self.world_size]

        if self.train_ratio > 0:
            return iter(train_indices)
        else:
            return iter(val_indices)

    def __len__(self):
        if self.train_ratio > 0:
            return int(len(self.dataset) * self.train_ratio) // self.world_size
        else:
            return int(len(self.dataset) * (1 - self.train_ratio)) // self.world_size

    def set_epoch(self, epoch):
        # 在每个epoch之前设置新的随机种子
        random.seed(epoch)




def train(rank, world_size,epochs,loss_rec):

    # 初始化进程组
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

    # CNN实例化
    model = Net()

    # 将模型移动到 CPU 上
    model = model.cpu()

    

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 定义数据和标签的 DataLoader
    train_sampler = DistributedRandomSplitSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=60, sampler=train_sampler)

    # 训练循环
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        num_batches = 0
        for input_data, target in train_loader:
            # 将数据和标签移到 CPU 上
            input_data = input_data.cpu()
            target = target.cpu()

            # 前向传播、计算损失和反向传播
            output = model(input_data)
            loss = criterion(output, target)
            # dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            # loss /= world_size
            optimizer.zero_grad()
            loss.backward()

            # 显式进行集合通信原语操作
            for param in model.parameters():
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            optimizer.step()

            # 累加损失
            total_loss += loss.item()
            num_batches += 1

        # 计算平均损失并存在Loss_rec里
        average_loss = total_loss / num_batches
        
        loss_rec[rank,epoch] = average_loss
        
        # 记录训练集的平均损失
        print(f'Epoch {epoch+1}/{epochs}, Rank {rank}: Train Loss: {average_loss}')
        


    if rank==0:
        torch.save(model,"model.plt")

    # 清理分布式训练环境
    dist.destroy_process_group()

if __name__ == '__main__':
    # 设置进程数量
    world_size = 2

    #设置训练轮数
    epochs=10

    #存储每一轮，每个rank的loss
    loss_rec = torch.zeros(world_size,epochs)

    # 使用 torch.multiprocessing 创建多个进程,并阻塞主进程
    mp.spawn(train, args=(world_size,epochs,loss_rec,), nprocs=world_size, join=True)
    

    #可视化loss_rec
    x = [i+1 for i in range(epochs)]
    y1 = loss_rec[0].tolist()
    y2 = loss_rec[1].tolist()

    # 创建画布和子图
    fig, ax = plt.subplots()

    # 绘制第一条loss折线
    ax.plot(x, y1, label='Loss-rank0')

    # 绘制第二条loss折线
    ax.plot(x, y2, label='Loss-rank1')

    # 添加图例
    ax.legend()

    # 添加标题和轴标签
    ax.set_title('loss comparision between rank0 and rank1')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')

    # 保存图形
    plt.savefig('loss1.png')



    
