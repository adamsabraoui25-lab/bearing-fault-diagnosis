import torch
from torch.utils.data import Dataset

class BearingDataset(Dataset):
    def __init__(self, X, y):
        # PyTorch attend une dimension "canal" pour les images : (N, 1, freq, temps)
        self.X = torch.from_numpy(X).unsqueeze(1).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]