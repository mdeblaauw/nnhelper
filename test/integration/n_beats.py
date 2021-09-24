from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, List


class NBeatsNet(nn.Module):

    def __init__(
        self,
        number_of_stacks: int,
        blocks_per_stack: int,
        forecast_length: int,
        backcast_length: int,
        thetas_dim: Tuple[int, int],
        hidden_layer_units: int
    ):
        """Univariate implementation of the NBeats model.

        Args:
            number_of_stacks (int): Number of sequential stacks.
            blocks_per_stack (int): The number of blocks per stack.
            forecast_length (int): The number of timesteps to forecast.
            backcast_length (int): The number of history timestaps to use.
            thetas_dim (Tuple[int, int]): The output length of the
                fully connected layer in every block, which is called the
                theta dimension in the paper.
            hidden_layer_units (int): The number of hidden neurons in the
                first four fully connected layers in a stack.
        """
        assert len(thetas_dim) == number_of_stacks, \
            (
                'The thetas_dim variable must have',
                ' length equal to number_of_stacks'
            )
        super().__init__()
        self.number_of_stacks = number_of_stacks
        self.blocks_per_stack = blocks_per_stack
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.thetas_dim = thetas_dim
        self.hidden_layer_units = hidden_layer_units

        self.stacks = nn.ModuleList([
            self._create_stack() for
            stack_number in
            range(self.number_of_stacks)
        ])

    def _create_stack(self) -> List[NBeatsBlock]:
        """Creates the blocks in a stack.

        Returns:
            List[NBeatsBlock]: List with blocks.
        """
        return nn.ModuleList([
            NBeatsBlock(
                units=self.hidden_layer_units,
                theta_dim=theta_dim,
                backcast_length=self.backcast_length,
                forecast_length=self.forecast_length
            ) for
            block_number, theta_dim in
            zip(range(self.blocks_per_stack), self.thetas_dim)
        ])

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Nbeats network.

        Returns:
            [type]: [description]
        """
        forecast = torch.zeros(size=(x.size()[0], self.forecast_length,))
        for stack in self.stacks:
            for block in stack:
                b, f = block(x)
                x = x - b
                forecast = forecast + f
        return forecast


class NBeatsBlock(nn.Module):

    def __init__(
        self,
        units: int,
        theta_dim: int,
        backcast_length: int,
        forecast_length: int,
    ):
        """An NBeats block that is part of a stack.

        Args:
            units (int): The number of hidden neurons in the
                first 4 fully connected layers.
            theta_dim (int): The number of output neurons for
                the theta layer.
            backcast_length (int): The number of history timesteps.
            forecast_length (int): The number of future timesteps.
        """
        super().__init__()
        self.units = units
        self.theta_dim = theta_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)

        self.theta_b_fc = nn.Linear(units, theta_dim, bias=False)
        self.theta_f_fc = nn.Linear(units, theta_dim, bias=False)

        self.backcast_fc = nn.Linear(theta_dim, self.backcast_length)
        self.forecast_fc = nn.Linear(theta_dim, self.forecast_length)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """The forward pass through a block.

        Returns:
            Tuple[torch.Tensor(), torch.Tensor()]: The estimation of the
                history and future.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        theta_b = self.theta_b_fc(x)
        theta_f = self.theta_f_fc(x)

        backcast = self.backcast_fc(theta_b)
        forecast = self.forecast_fc(theta_f)

        return backcast, forecast
