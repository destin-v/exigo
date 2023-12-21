import subprocess

import numpy as np
import torch
from rich import print
from rich.panel import Panel
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from src.data import get_mnist_dataloaders
from src.eval import eval_iteration
from src.model import BasicCNN
from src.plot import generate_projection
from src.plot import plot_confusion_matrix
from src.plot import saliency_map
from src.train import generate_canocial
from src.train import training_iteration

# set global seed
torch.manual_seed(0)


def run_pipeline(
    model: nn.Module,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    dataset_train: Dataset,
    dataset_val: Dataset,
    n_epochs: int,
    min_loss: float,
    writer: SummaryWriter,
    model_weights_path: str,
):
    """The full training pipeline performs a loop over the following:

    ```Loop
    1) Train the model
    2) Perform evaluation
    3) Record the average loss
        If the model has the best average loss:
            Save off the model
            Generate a confusion matrix
        Else:
            continue training
    ```

    .. note::
        The `dataloaders` must provide `(x, y)` representing inputs and true labels.

    Args:
        model (nn.Module): Torch model.
        dataloader_train (DataLoader): DataLoader with `(x, y)` data.
        dataloader_val (DataLoader): DataLoader with `(x, y)` data.
        dataset_train (Dataset): Datset with `(x, y)` data.
        dataset_val (Dataset): Datset with `(x, y)` data.
        n_epochs (int): Number of epochs to run training.
        min_loss (float): Minimium loss recorded for the training iterations.
        writer (SummaryWriter): Tensorboard writer.
        model_weights_path (str): Path to save off model.
    """
    # Create optimizer and loss functions
    optimizer = Adam(params=model.parameters(), lr=0.00001)
    loss_fn = CrossEntropyLoss()

    # Perform training on the data
    for global_step in range(0, n_epochs):
        print(f"Global step: {global_step}")

        # Perform training
        avg_train_loss = training_iteration(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            dataloader_train=dataloader_train,
            global_step=global_step,
            writer=writer,
        )

        # Perform evaluation
        avg_eval_loss = eval_iteration(
            model=model,
            loss_fn=loss_fn,
            dataloader_val=dataloader_val,
            global_step=global_step,
            writer=writer,
        )

        # Printout
        print(f"Epoch:               {global_step}/{n_epochs-1}")
        print(f"    Training   Loss: {avg_train_loss:.3f}")
        print(f"    Evaluation Loss: {avg_eval_loss:.3f}")

        # Save off best model
        if avg_eval_loss < min_loss:
            # Update the min_loss as the new baseline
            min_loss = avg_eval_loss

            # Save off the best model
            torch.save(model.state_dict(), model_weights_path)

            # Run analysis
            run_analysis(
                model=model,
                dataloader_val=dataloader_val,
                dataset_val=dataset_val,
                global_step=global_step,
                writer=writer,
            )

            # Printout
            print(f"Saving best model with evaluation loss of: {avg_eval_loss:.2f}")


def run_analysis(
    model: nn.Module,
    dataloader_val: DataLoader,
    dataset_val: Dataset,
    global_step: int,
    writer: SummaryWriter,
):
    """Run analysis on each epoch where the model improves.  This should log multiple metrics that can be used in post-analysis examination.

    Args:
        model (nn.Module): Torch model.
        dataloader_val (DataLoader): Dataloader for validation.
        dataset_val (Dataset): Datset for validation.
        global_step (int): Global step for tracking.
        writer (SummaryWriter): Tensorboard writer.
    """

    # Save off confusion matrix
    plot_confusion_matrix(
        model=model,
        dataloader_val=dataloader_val,
        save_path="save/plots/confusion_matrix",
        fig_name=f"confusion_matrix_{global_step}",
    )

    # Generate projection logs
    generate_projection(
        model=model,
        dataloader_val=dataloader_val,
        writer=writer,
        global_step=global_step,
    )

    # Generate saliency plot
    saliency_map(
        model=model,
        dataset_val=dataset_val,
        figure_name="saliency",
        global_step=global_step,
        save_path=f"save/plots/saliency",
    )


def run_mnist(
    n_epochs: int = 2,
    min_loss: float = np.inf,
    batch_size: int = 32,
    path_model: str = "save/models",
    path_log: str = "save/logs",
    path_plots: str = "save/plots",
    model_name: str = "best_model.pkl",
):
    """Perform MNIST training.  This function performs the following:

        1) Create save/log directories
        2) Get data iterators for train and evaluation
        3) Create Tensorboard writer
        4) Create model
        5) Execute training pipeline using the data iterators, Tensorboard writer, and model provided

    Args:
        n_epochs (int, optional): Number of epochs to perform training. Defaults to 2.
        min_loss (float, optional): The minimum loss recorded thus far. Defaults to np.inf.
        batch_size (int, optional): The batch size to use when retrieving data from data iterators. Defaults to 16.
        path_model (str, optional): The path to the model. Defaults to "save/models".
        path_log (str, optional): The path to the logs. Defaults to "logs".
        model_name (str, optional): The name of the model. Defaults to "best_model.pkl".
    """
    # Fully defined path to model save file
    model_weights_path = f"{path_model}/{model_name}"

    # Create directories
    subprocess.run(["mkdir", "-p", path_model])
    subprocess.run(["mkdir", "-p", path_log])
    subprocess.run(["mkdir", "-p", path_plots])

    # Preparing pipeline
    dataloader_train, dataloader_val, dataset_train, dataset_val = get_mnist_dataloaders(
        batch_size=batch_size
    )
    writer = SummaryWriter(path_log)  # Tensorboard
    model = BasicCNN()  # hyperparameters

    # Preprocessing
    saliency_map(
        model,
        dataset_val=dataset_val,
        figure_name="saliency",
        global_step=-1,
        save_path=f"{path_plots}/saliency",
    )

    # Save off model architecture.  Requires that you pass in a torch tensor for forward pass in order for logging to happen.
    writer.add_graph(model, torch.rand(1, 1, 28, 28))

    # Train the model
    run_pipeline(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        dataset_train=dataset_train,
        dataset_val=dataset_val,
        n_epochs=n_epochs,
        min_loss=min_loss,
        writer=writer,
        model_weights_path=model_weights_path,
    )

    # TODO: generate_canocial is not currently working
    writer.close()

    # Printout
    print(Panel("To open Tensorboard type:"))
    print("     tensorboard --logdir=save/logs")
    print("     :warning: Chrome is recommended for viewing projections!")


if __name__ == "__main__":
    run_mnist()
