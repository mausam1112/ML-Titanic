import torch
from torch.utils.data import Dataset, DataLoader
from core.config import Configs


class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(X, y, train=True):
    dataset = TitanicDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=Configs.batch_size, shuffle=train)

    return dataloader
