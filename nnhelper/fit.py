import torch
from typing import Callable
from .callbacks.base_callback import CallbackList, Callback
from .callbacks.metric_aggregator import MetricAggregator


def fit(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fit_function: Callable = standard_fit_function,
    callbacks: Callback = None,
    metrics=[],
    verbose: int = 1,
    device: str = 'cpu',
    start_epoch: int = 1
):
    """[summary]

    Args:
        model ([type]): [description]
        fit_function ([type]): [description]
        data_loader ([type]): [description]
        epochs (int): [description]
        loss_fn ([type]): [description]
        optimizer ([type]): [description]
        callbacks ([type], optional): [description]. Defaults to None.
        metrics (list, optional): [description]. Defaults to [].
        verbose (int, optional): [description]. Defaults to 1.
        device (str, optional): [description]. Defaults to 'cpu'.
        start_epoch (int, optional): from where to start epochs.
            This is normally only necessary when to restart from
            a checkpoint. Defaults to 1.
    """
    num_batches = len(data_loader)

    batch_size = data_loader.batch_size

    callbacks = CallbackList([MetricAggregator()] + (callbacks or []))
    callbacks.set_model(model)
    callbacks.set_params({
        'optimizer': optimizer,
        'num_batches': num_batches,
        'batch_size': batch_size,
        'metrics': metrics,
        'verbose': verbose
    })

    callbacks.on_train_begin()

    for epoch in range(start_epoch, epochs+1):
        callbacks.on_epoch_begin(epochs)

        epoch_logs = {}
        for batch_index, batch in enumerate(data_loader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            def handle_batch(batch_logs, batch):
                """Preprocess batch, add it to device, and
                apply forward and backward pass.

                Args:
                    batch_logs ([type]): [description]
                    batch ([type]): [description]
                """
                x, y = batch[0].to(device), batch[1].to(device)

                callbacks.on_batch_begin(batch_index, batch_logs)

                losses = fit_function(model, (x, y), loss_fn, optimizer)

                for loss in losses:
                    batch_logs[loss] = losses[loss].item()

            handle_batch(batch_logs, batch)

            callbacks.on_batch_end(batch_index, batch_logs)

        # run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # run on train end
    callbacks.on_train_end()


def standard_fit_function(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> Dict[str, float]:
    """[summary]

    Args:
        model (torch.nn.Module): [description]
        input_batch (torch.Tensor): [description]
        loss_fn (torch.nn.Module): [description]
        optimizer (torch.optim.Optimizer): [description]

    Returns:
        Dict[str, float]: [description]
    """
    model.train()
    optimizer.zero_grad()

    pred = model(input_batch[0])

    losses = loss_fn(input_batch[1], pred)

    losses['loss'].backward()
    optimizer.step()

    return losses