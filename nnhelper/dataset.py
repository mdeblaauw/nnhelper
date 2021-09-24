import torch
import numpy as np
import torch.utils.data as data
from typing import List, Union, Tuple


class Dataset(data.Dataset):
    """A simple Pytorch Dataset class for a DataLoader.
    """
    def __init__(
        self,
        x: List[Union[float, np.ndarray, torch.Tensor]],
        y: List[Union[float, np.ndarray, torch.Tensor]]
    ):
        """Initialise the Dataset with train data and target data.

        Args:
            x (List[Union[float, np.darray, torch.Tensor]]):
                The train data.
            y (List[Union[float, np.darray, torch.Tensor]]):
                The target data
        """
        assert type(x) == type(y), \
            'Both `x` and `y` must be the same type'

        if(isinstance(x, float)):
            self.x = [
                torch.tensor(sample, dtype=torch.float) for
                sample in x
            ]
            self.y = [
                torch.tensor(target, dtype=torch.float) for
                target in y
            ]
        elif(isinstance(x, np.ndarray)):
            self.x = [
                torch.from_numpy(sample).type(torch.float) for
                sample in x
            ]
            self.y = [
                torch.from_numpy(target).type(torch.float) for
                target in y
            ]
        else:
            self.x = x
            self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples train and target data.

        Args:
            index (int): Which sample to take.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Train and target sample.
        """
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
