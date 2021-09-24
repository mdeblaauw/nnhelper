# nnhelper
Implementing neural networks from papers or experimenting with own ideas often includes writing some boiler plate code. You can think of the `epoch for loop`, `evaluation round`, and `saving` the model. This package has some standard functions so that you can focus all your time on modelling and experimenting.

For now, the package is only tested and is focused on neural networks with Pytorch. Hence, the preferred language is Python.

What else can you find in this readme:

* Install and test the package.
* An example of how to use this package.
* Add additional functionality.
* FAQ.

## Install and test the package
This package is tested and developed with Python 3.9. So, make sure you have Python3 installed and version 3.9 or higher.

Create a virual Python environment:

`python3 -m venv venv`

`source venv/bin/activate`

In the MakeFile you can run the tests:

`make test`

To only install the `nnhelper` package into your Python environment you can do this:

`pip install git+https://github.com/mdeblaauw/nnhelper.git`

## An example of how to use this package

```python
import math
import torch
import nnhelper
import nnhelper.callbacks as callbacks

# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
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
    epochs=10,
    loss_fn=loss_fn,
    optimizer=optimizer,
    callbacks=[
        callbacks.VerboseLogger()
    ]
)
```

## Add additional functionality
There is a large versatility in neural network architectures. It could therefore occur that the `Dataset` function, `fit function`, or `loss` function is not able to meet your architecture requirements. This package is created so that you can easily add more complex functionalities and bring them along in the `nnhelper.fit` function.

## FAQ