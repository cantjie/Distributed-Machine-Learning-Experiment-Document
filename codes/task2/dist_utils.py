import os
import torch
import torch.distributed as dist 


def dist_init(world_size, rank, master_addr='localhost', master_port='12355'):
    # change it to the corresponding ip addr
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # initialize the process group
    print("HMM inside 0")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    assert dist.is_initialized(), "Error! The distributed env is not initialized!"
    print("HMM inside 1")

    return True


def cleanup():
    dist.destroy_process_group()


def get_local_rank():
    # get the local rank (devices id)
    if not dist.is_initialized():
        return 0
    else:
        return dist.get_rank()


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()


def init_parameters(model):
    # Boradcast the initial gradients of the model parameters
    if get_world_size() > 1:
        for param in model.parameters():
            dist.broadcast(param.data,0)


def allreduce_average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        # your code here
        # implement your own aggregation method
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


def allgather_average_gradients(model):
    # your code here
    gradients = [torch.Tensor(param.grad.data.clone()) for param in model.parameters() if param.grad is not None]
    # gradients = torch.Tensor(gradients )
    all_gradients = [torch.zeros_like(gradient) for gradient in gradients]
    dist.all_gather(all_gradients, gradients)

    num_processes = float(dist.get_world_size())
    for gradient in all_gradients:
        gradient.div_(num_processes)
    
    for param, gradient in zip(model.parameters(), all_gradients):
        if param.grad is not None:
            param.grad.data = gradient