import torch
from .base_callback import Callback


class SaveAsPytorch(Callback):
    """Callback that stores the model into a .pt extension with
        Pytorch's state_dict().
    """
    def __init__(
        self,
        path: str,
        alternative_model=None,
        save_every_epoch: bool = False,
        batch_iterations: int = 0
    ):
        """[summary]

        Args:
            path (str): Path + filename to store the Pytorch model. E.g., `<path>/model.pt`.
            alternative_model ([type], optional): [description]. Defaults to None.
            save_every_epoch (bool, optional): [description]. Defaults to False.
            batch_iterations (int, optional): [description]. Defaults to 0.
        """
        self.batch_iterations = batch_iterations
        self.path = path
        self.save_every_epoch = save_every_epoch
        if alternative_model:
            self.model_to_save = alternative_model
        else:
            self.model_to_save = self.model

    def on_batch_begin(self, batch, logs=None):
        self.batch_iterations += 1

    def on_epoch_end(self, epoch: int, logs=None):
        if self.save_every_epoch:
            if isinstance(self.model_to_save, dict):
                checkpoint_dir = {
                    'epoch': epoch,
                    'batch_iterations': self.batch_iterations,
                    'optimizer_state_dict': self.params['optimizer'].state_dict()
                }

                for model_name in self.model_to_save:
                    checkpoint_dir[f'{model_name}_state_dict'] = self.model_to_save[model_name].state_dict()

                torch.save(
                    checkpoint_dir,
                    f'{self.path}_checkpoint_epoch={epoch}.pth'
                )
            else:
                torch.save(
                    {
                        'epoch': epoch,
                        'batch_iterations': self.batch_iterations,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.params['optimizer'].state_dict(),
                    },
                    f'{self.path}_checkpoint_epoch={epoch}.pth'
                )

    def on_train_end(self, logs=None):
        if isinstance(self.model_to_save, dict):
            for model_name in self.model_to_save:
                torch.save(
                    self.model_to_save[model_name].state_dict(),
                    f'{self.path}_{model_name}.pth'
                )
        else:
            torch.save(
                self.model_to_save.state_dict(),
                f'{self.path}.pth'
            )