import numpy as np
import torch
from rich.progress import track
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


def eval_iteration(
    model: nn.Module,
    loss_fn: nn.Module,
    dataloader_val: DataLoader,
    global_step: int,
    writer: SummaryWriter,
) -> float:
    """Perform evaluation on the model using the DataLoader.

    Args:
        model (nn.Module): Torch model.
        loss_fn (nn.Module): Torch loss.
        dataloader_val (DataLoader): A DataLoader that provides `(x, y)` where `x` is the input data and `y` is the true label.
        global_step (int): The global step for tracking the evaluation.
        writer (SummaryWriter): Tensorboard writer.

    Returns:
        float: Average evaluation loss.
    """
    # Initialization
    eval_losses = []

    # evaluation iteration
    for x, y in track(dataloader_val, description="evaluation"):
        # Turn off gradients when calculating loss
        with torch.no_grad():
            y_hat, _ = model(x)
            loss = loss_fn(y_hat, y.long())
            eval_losses.append(loss.item())

    # Calculate the average loss and record it
    avg_eval_loss = float(np.average(eval_losses))
    writer.add_scalar(tag="Loss/eval", scalar_value=avg_eval_loss, global_step=global_step)

    return avg_eval_loss
