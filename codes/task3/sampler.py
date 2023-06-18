import math
import torch
from torch.utils.data import Dataset, Sampler
import random
import torch.distributed as dist

class MySampler(Sampler):
    def __init__(self, dataset:Dataset, num_replicas, rank, shuffle=True, seed=0):
        super(Sampler, self).__init__()
        self.dataset = dataset
        self.num_replicas = num_replicas    # number of clients (processes)
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed                    # set seed to be the rank of the client, to avoid generating the same indice lists.
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas) 

    def __iter__(self):
        """
            example:
                indices=list(range(len(self.dataset)))
                return iter(indices)
        """
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Set the random seed based on rank and epoch to ensure different indices across different processes
            seed = self.seed + self.epoch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            indices = torch.randperm(len(indices)).tolist()

        indices = indices[self.rank:len(indices):self.num_replicas]  # Select samples for the current rank

        return iter(indices)

    def __len__(self):
        return self.num_samples
