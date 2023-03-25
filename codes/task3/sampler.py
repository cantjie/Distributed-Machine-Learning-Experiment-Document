import math
import torch
from torch.utils.data import Dataset, Sampler

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
        raise NotImplementedError

    def __len__(self):
        return self.num_samples
