import numpy as np
from rich import print
from torch.nn import CrossEntropyLoss
from torch.optim.adam import Adam
from torch.utils.tensorboard.writer import SummaryWriter

from src.data import get_mnist_dataloaders
from src.model import BasicCNN
from src.train import training_iteration


def test_training_iteration(tmpdir):
    """Exercise the training iteration to ensure that everything performs as expected without errors."""

    # Setup Dataloader and writers
    _, dataloader_val, _, _ = get_mnist_dataloaders()
    log_dir = str(tmpdir)
    writer = SummaryWriter(log_dir=log_dir)

    # Create a model and learning parameters
    model = BasicCNN()
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Perform two training iterations to exercise the functionality
    min_loss = np.inf
    for global_step in range(0, 2):
        average_loss = training_iteration(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader_train=dataloader_val,
            global_step=global_step,
            writer=writer,
        )

        # Verify loss is decreasing
        print(f"Epoch:          {global_step}")
        print(f"    Avg Loss:   {average_loss}")
        assert average_loss < min_loss, "Loss is not converging!"  # type: ignore

        # Update the min loss
        if average_loss < min_loss:  # type: ignore
            min_loss = average_loss
