import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np


class Next_word_pred_dataset(Dataset):

    def __init__(self, dataset_ndarray, vocab_size, device='cpu'):
        # dataset_ndarray is expected to be of shape [batch_size, seq_len]
        # Last word in the seq is taken as prediction

        self.x_words = torch.tensor(dataset_ndarray[:, :-1], dtype=torch.int64).to(device)
        self.y_next_word = torch.tensor(dataset_ndarray[:, -1], dtype=torch.int64).to(device)
        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        y = F.one_hot(self.y_next_word[idx], self.vocab_size).to(torch.float64)
        return self.x_words[idx, :], y

    def __len__(self):
        return self.x_words.shape[0]

    def get_data_loader(self, batch_size=250, shuffle=True):
        return DataLoader(self, batch_size=batch_size, drop_last=True, shuffle=shuffle) #We drop the last batch if it is small, to avoid error in batch_norm layer
