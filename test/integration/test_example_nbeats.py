import unittest
import pandas as pd
import torch
import numpy as np
import nnhelper
import nnhelper.callbacks as callbacks
from test.integration.n_beats import NBeatsNet


class TestNnhelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        milk_data = pd.read_csv(
            'test/data/milk.csv',
            index_col=0,
            parse_dates=True
        )
        cls.milk = milk_data.values.flatten()

    def setUp(self):
        self.epochs = 10
        self.forecast_length = 3
        self.backcast_length = 3 * self.forecast_length
        self.batch_size = 10

        # data backcast/forecast generation.
        x, y = [], []
        for self.epoch in range(
            self.backcast_length,
            len(self.milk) - self.forecast_length
        ):
            x.append(self.milk[self.epoch - self.backcast_length:self.epoch])
            y.append(self.milk[self.epoch:self.epoch + self.forecast_length])
        x = np.array(x)
        y = np.array(y)

        # split train/test.
        c = int(len(x) * 0.8)
        x_train, y_train = x[:c], y[:c]
        x_test, y_test = x[c:], y[c:]

        # normalization.
        norm_constant = np.max(x_train)
        x_train, y_train = x_train / norm_constant, y_train / norm_constant
        x_test, y_test = x_test / norm_constant, y_test / norm_constant

        self.data_loader = torch.utils.data.DataLoader(
            dataset=nnhelper.Dataset(x=x_train, y=y_train),
            batch_size=self.batch_size
        )

    def test_fit_function(self):
        # Create Model
        model = NBeatsNet(
            number_of_stacks=2,
            blocks_per_stack=3,
            forecast_length=self.forecast_length,
            backcast_length=self.backcast_length,
            thetas_dim=(4, 8),
            hidden_layer_units=128
        )

        # Create loss function
        loss_fn = torch.nn.MSELoss()

        # Make optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-4
        )

        # fit the model on the train data
        nnhelper.fit(
            model=model,
            data_loader=self.data_loader,
            epochs=self.epochs,
            loss_fn=loss_fn,
            optimizer=optimizer,
            callbacks=[
                callbacks.VerboseLogger()
            ]
        )
