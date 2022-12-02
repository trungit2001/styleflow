import numpy as np
from torch.utils.data.sampler import Sampler

from models.dataset import ImagePairDataset

class DistributedGivenIterationSampler(Sampler):
    def __init__(
        self,
        dataset: ImagePairDataset,
        total_iter: int,
        batch_size: int,
        world_size=None,
        rank=None,
        last_iter: int = -1
    ) -> None:
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size
        self.indices = self.gen_new_list()
        self.call = True

    def __iter__(self):
        if self.call:
            self.call = False
            return iter(self.indices[(self.last_iter + 1) * self.batch_size:])
        else:
            raise RuntimeError("This sampler is not designed to be called more than once!!")

    def gen_new_list(self):
        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        begin = self.total_size * self.rank
        indices = indices[begin:begin + self.total_size]
        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        return self.total_size
