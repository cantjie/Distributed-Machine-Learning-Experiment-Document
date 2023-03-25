from mindspore import nn, ops
# from mindspore import Parameter, Tensor
# import mindspore as ms

class GdOptimizer(nn.Optimizer):
    def __init__(self, params, lr=0.001):
        super(GdOptimizer, self).__init__(lr, params)
    
    def construct(self, gradients):
        raise NotImplementedError
    

class AdamOptimizer(nn.Optimizer):
    pass