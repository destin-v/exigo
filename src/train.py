import matplotlib.pyplot as plt
import numpy as np
import torch
from rich import print
from rich.progress import track
from torch import nn
from torch.nn import MSELoss
from torch.optim import Optimizer
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter


def training_iteration(
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    dataloader_train: DataLoader,
    global_step: int,
    writer: SummaryWriter,
) -> np.floating:
    """Perform a single training iteration on the DataLoader.  Average losses are recorded and saved to Tensorboard logs.

    Args:
        model (nn.Module): Torch model.
        loss_fn (nn.Module): Torch loss.
        optimizer (Optimizer): Torch optimizer.
        dataloader_train (DataLoader): DataLoader that outputs `(x, y)` where `x` is input and `y` is the true label.
        global_step (int): The global step for tracking this training iteration.
        writer (SummaryWriter): Tensorboard writer.

    Returns:
        np.floating: Average training loss for this iteration.
    """
    # Initialization
    train_losses = []

    for x, y in track(dataloader_train, description="training"):
        # fmt: off
        y_hat, _ = model(x)                 # predict
        loss = loss_fn(y_hat, y.long())     # calculate loss
        optimizer.zero_grad()               # zero the gradients
        loss.backward()                     # backpropagation
        optimizer.step()                    # update the weights
        train_losses.append(loss.item())    # save off the loss
        # fmt: on

    # Record the average training loss for this training iteration
    avg_train_loss = np.average(train_losses)
    writer.add_scalar(tag="Loss/train", scalar_value=avg_train_loss, global_step=global_step)

    return avg_train_loss


def generate_canocial(
    model: nn.Module,
    model_weights_path: str,
    n_epochs: int = 10,
    n_steps: int = 100,
):
    r"""The idea behind a canocial input is that it should be the input that maximizes the predicted value of a network.  Think of this as `inverse` training rather than standard training.  You want to specify an output and ask the network what is the corresponding input.

    The way to achieve this is by freezing all of the layers of a network and setting $y_{true}$ to your target class.  When performing backpropagation via $loss=\hat{y} - y_{true}$ you backpropagate the gradients all the way to the input.

    The theory is that once the losses converge to zero, the input should represent the maximum values needed to push the predictions of your neural network to the maximum for a certain class.

    Args:
        model (nn.Module): Torch model
        model_weights_path (str): Path to model saved weights.
        n_epochs (int, optional): Number of epochs to perform training. Defaults to 10.
        n_steps (int, optional): Number of steps within an epoch. Defaults to 100.
    """

    # Create an image to start inverse training
    input = torch.rand((1, 28, 28))  # assume 28x28 image
    input.requires_grad = True  # set to make sure input can be adjusted

    # Create embedding visualization
    model.load_state_dict(torch.load(model_weights_path))

    # Freeze all parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    # Setup optimizer
    optimizer = SGD([input], lr=0.001)
    loss_fn = MSELoss()

    target = torch.Tensor([10, -10, -10, -10, -10, -10, -10, -10, -10, -10])
    images = []

    # Perform training epochs
    for ii in range(0, n_epochs):
        losses = []
        for jj in range(0, n_steps):
            pred, _ = model(input)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        image = input.squeeze().clone().detach().numpy()
        images.append(image)
        print(f"Generative loss: {np.mean(losses)}")

    # Draw images
    for ii in range(1, len(images)):
        ax = plt.subplot(1, len(images), ii)
        ax.imshow(images[ii])
