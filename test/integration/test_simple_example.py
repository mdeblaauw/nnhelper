import unittest
import math
import torch
import nnhelper
import nnhelper.callbacks as callbacks


class TestReadmeExample(unittest.TestCase):
    def test_running_readme_example(self):
        # Create Tensors to hold input and outputs.
        x = torch.linspace(-math.pi, math.pi, 100)
        y = torch.sin(x)

        p = torch.tensor([1, 2, 3])
        xx = x.unsqueeze(-1).pow(p)

        # Only 1 sample. Hence, batch size of 1.
        data_loader = torch.utils.data.DataLoader(
            dataset=nnhelper.Dataset(x=xx, y=y),
            batch_size=1
        )

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.Flatten(0, 1)
        )

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-2
        )

        # Train the model with 10 epochs and use
        # Verbose Logger to put metrics to terminal.
        nnhelper.fit(
            model=model,
            data_loader=data_loader,
            epochs=2,
            loss_fn=loss_fn,
            optimizer=optimizer,
            callbacks=[
                callbacks.VerboseLogger()
            ]
        )
