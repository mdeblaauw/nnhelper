import torch
import torch.utils.data as data
from typing import List


class Dataset(data.Dataset):

    def __init__(self, x: List[float], y: List[float]):
        self.x = x
        self.y = y

    def __getitem__(self, index) -> torch.Tensor():
        return (
            torch.tensor(self.x[index], dtype=torch.float),
            torch.tensor(self.y[index], dtype=torch.float)
        )

    def __len__(self):
        return len(self.x)