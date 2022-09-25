import torch
from collections import defaultdict


class Sampler:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset_size, self.batch_size, self.shuffle = (
            len(dataset),
            batch_size,
            shuffle,
        )

    def __iter__(self):
        self.indices = (
            torch.randperm(self.dataset_size)
            if self.shuffle
            else torch.arange(self.dataset_size)
        )
        for i in range(0, self.dataset_size, self.batch_size):
            yield self.indices[i : i + self.batch_size]


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.sampler = Sampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self):
        return len(self.dataset)

    def collate_fn(self, batch):
        collated_batch = defaultdict(list)
        for sample in batch:
            for column in batch[0].keys(): 
                collated_batch[column].append(sample[column])
        collated_batch = {k: torch.stack(v) for k, v in collated_batch.items()}
        return collated_batch

    def __iter__(self):
        for batch_indices in self.sampler:
            batch = [self.dataset[i] for i in batch_indices]
            yield self.collate_fn(batch)
