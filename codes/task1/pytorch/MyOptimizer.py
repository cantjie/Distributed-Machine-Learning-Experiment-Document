from torch.optim import SGD, Adam

import torch

#TODO: now

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.predict(x)             
        return x

criterion = torch.nn.MSELoss()


#  Gradient Descent (GD):
net_all = Net(n_feature=1, n_hidden=10, n_output=1) 
optimizer_all = torch.optim.SGD(net_all.parameters(), lr=0.2)

net_item = Net(n_feature=1, n_hidden=10, n_output=1) 
optimizer_item = torch.optim.SGD(net_item.parameters(), lr=0.2)

class BaseOptimizer():
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
    
    def step(self):
        raise NotImplementedError
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()


class GdOptimizer(BaseOptimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)
        
    def step(self):
        for p in self.params:
            if p.grad is None:
                # your code here
                pass


class AdamOptimizer(BaseOptimizer):
    pass