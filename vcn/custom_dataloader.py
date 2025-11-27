#!/usr/bin/env python3
import numpy as np


class MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.num_batches = self.data_size // self.batch_size
        if self.data_size % self.batch_size > 0:
            self.num_batches += 1
        self.ids = np.arange(self.data_size)
        self.current_iter = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.ids)
        self.current_iter = 0
        return self

    def __next__(self):
        if self.current_iter >= self.num_batches:
            raise StopIteration
        start = self.current_iter * self.batch_size
        end = start + self.batch_size
        if end > self.data_size:
            end = self.data_size
        self.current_iter += 1
        return self.dataset[self.ids[start:end]]
