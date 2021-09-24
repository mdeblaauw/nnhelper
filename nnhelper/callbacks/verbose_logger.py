import torch
from .base_callback import Callback


class VerboseLogger(Callback):
    """Callback that prints training information to the terminal.
       For example, epoch number and corresponding average loss.
    """
    def __init__(self, log_interval: int = 1):
        """[summary]

        Args:
            log_interval (int, optional): If verbose is 2, then this sets the number of
                images after which the training loss is logged. Defaults to 1.
        """
        self.batch_iterations = 0
        self.log_interval = log_interval

    def on_batch_begin(self, batch, logs=None):
        if self.params['verbose'] == 2:
            self.batch_iterations += 1
            if(self.batch_iterations % self.log_interval == 0):
                print('Batch:', self.batch_iterations)

    def on_batch_end(self, batch, logs=None):
        if self.params['verbose'] == 2:
            if(self.batch_iterations % self.log_interval == 0):
                for k in self.metrics:
                    if logs[k]:
                        print(f'{k}:{logs[k]}')

    def on_epoch_end(self, epoch, logs=None):
        if self.params['verbose'] > 0:
            print('Epoch:', epoch)
            for k in self.metrics:
                if logs[k]:
                    print(f'{k}:{logs[k]}')

    def on_train_begin(self, logs=None):
        self.metrics = ['loss'] + self.params['metrics']
        if self.params['verbose'] > 0:
            print('Begin training...')

    def on_train_end(self, logs=None):
        if self.params['verbose'] > 0:
            print('Finished.')