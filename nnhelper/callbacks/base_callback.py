from __future__ import annotations
import torch
from typing import List, Dict, Any


class CallbackList:
    """Container abstracting a list of callbacks
    """
    def __init__(self, callbacks: List[Callback]):
        """Provide a list of callbacks to be used.

        Args:
            callbacks (List[Callback]): Callback classes that
                inherit the Callback base class.
        """
        self.callbacks = [c for c in callbacks]

    def set_params(self, params: Dict[str, Any]):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model: torch.nn.Module):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_train_begin(self, logs=None):
        """
        Called at the beginning of training
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """
        Called at the end of training
        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """
        Called at the beginning of an epoch
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of an epoch
        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """
        Called at the start of a batch
        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """
        Called at the end of a batch
        # Arguments
            batch: integer, index of batch whithin current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class Callback:
    """Base class that every Callback must inherit.
    """
    def __init__(self):
        self.model = None

    def set_params(self, params: Dict[str, Any]):
        self.params = params

    def set_model(self, model: 'torch.nn.Module'):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass
