import subprocess
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter


def plot_confusion_matrix(
    model: nn.Module,
    dataloader_val: DataLoader,
    save_path: str = "save",
    fig_name: str = "",
    fig_x_size: int = 8,
    fig_y_size: int = 8,
    dpi: int = 250,
):
    """Given a model and a DataLoader (which provides inputs and labels), evaluate how well the model makes predictions on the data.  We compare the highest predicted value against the true label and generate a confusion matrix.

    Args:
        model (nn.Module): Torch model.
        dataloader_val (DataLoader): DataLoader that outputs `(x, y)`.
        save_path (str, optional): Path to save plots. Defaults to "save".
        fig_name (str, optional): Figure name. Defaults to "".
        fig_x_size (int, optional): Figure x size. Defaults to 8.
        fig_y_size (int, optional): Figure y size. Defaults to 8.
        dpi (int, optional): Increase for higher resolution. Defaults to 250.
    """

    # Make dirs
    subprocess.run(["mkdir", "-p", save_path])

    # Preallocate lists
    y_pred = []
    y_true = []

    # Iterate over all inputs and labels for the data iterator
    for x, y in dataloader_val:
        # Perform a forward pass with the input data x and get highest predicted value
        y_hat, _ = model(x)
        y_idx = torch.argmax(y_hat, 1)

        # Add the prediction and true label to the preallocated list
        y_pred.extend(list(y_idx.numpy()))
        y_true.extend(list(y.numpy()))

    # Generate the confusion matrix using the predictions versus the true labels
    cm = confusion_matrix(y_true, y_pred)

    # Generate a figure and plot the results
    figure = plt.figure(figsize=(fig_x_size, fig_y_size))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Add axes labels
    plt.title("Confusion Matrix")
    plt.ylabel("Predictions")
    plt.xlabel("Truth")

    # Save off the plot
    plt.savefig(f"{save_path}/{fig_name}.png", dpi=dpi)
    plt.close("all")


def generate_projection(
    model: nn.Module,
    dataloader_val: DataLoader,
    writer: SummaryWriter,
    global_step: int = 0,
    projection_limit: int = 1000,
):
    """Create a projection of the data in N-dimensional space.  This can be visualized using Tensorboard.

    Args:
        model (nn.Module): Torch model.
        dataloader_val (DataLoader): Dataloader that outputs `(x, y)`.
        writer (SummaryWriter): Tensorboard writer.
        global_step (int, optional): The global step tracker. Defaults to 0.
        projection_limit (int, optional): The maximum number of projections allowed. Defaults to 1000.
    """
    # Initialize variables
    embeddings = []
    labels = []
    images = []

    # Iterate over all data
    for x, y in dataloader_val:
        # Remove gradient tape
        with torch.no_grad():
            _, embedding = model(x)  # get the embeddings (not predictions of the model)
            embeddings.append(embedding)  # save off embeddings

        # Append the data
        labels.append(y)
        images.append(x)

    # Stack the output embeddings
    stacked_embeddings = torch.vstack(embeddings)
    stacked_labels = torch.hstack(labels).numpy()  # convert to numpy array instead of tensors
    stacked_images = torch.vstack(images)

    # Tensorboard is only able to plot a maximum number of data points in its projection.  Thus, we will limit the number of data points we will display.
    mean, std, _ = (
        torch.mean(stacked_embeddings),
        torch.std(stacked_embeddings),
        torch.var(stacked_embeddings),
    )
    stacked_embeddings = (stacked_embeddings - mean) / std
    stacked_embeddings = stacked_embeddings[0:projection_limit, :]
    stacked_labels = stacked_labels[0:projection_limit]
    stacked_images = stacked_images[0:projection_limit, :, :, :]

    # Add the embedding information to Tensorboard logs
    writer.add_embedding(
        stacked_embeddings,
        metadata=stacked_labels,
        # label_img=stacked_images,     # turn on to display images for each data point!
        global_step=global_step,
    )


def saliency_map(
    model: nn.Module,
    dataset_val: Dataset,
    figure_name: str = "saliency",
    global_step: int = 0,
    model_weights_path: str | None = None,
    save_path: str = "save",
):
    """Create a saliency map showing which inputs contributed the most toward the predictions.

    Args:
        model (nn.Module): Torch model.
        dataset_val (Dataset): The Dataset which provides `(x, y)`.
        figure_name (str): The name of the figure to save as file. Defaults to "saliency".
        global_step (int): The global step for tracking. Defaults to 0.
        model_weights_path (str | None): Torch model weights path.
        save_path (str): Save path.
    """

    # Create dirs if they do not exists
    subprocess.run(["mkdir", "-p", save_path])

    # configuration (must be even number!)
    n_rows = 5
    n_cols = 10

    # Copy model
    nn_model = deepcopy(model)
    if model_weights_path is not None:  # Load weights into model if available
        nn_model.load_state_dict(torch.load(model_weights_path))

    data_iter = iter(dataset_val)
    data_idx = 0

    for ii in range(0, n_rows, 5):
        for jj in range(1, n_cols + 1, 1):
            x, y = next(data_iter)
            data_idx += 1

            # Calculate gradients w.r.t. input from output
            x_base = torch.ones((1, 28, 28)) * -1
            x_pred = x

            x_base.requires_grad = True  # set gradient tape to True
            x_pred.requires_grad = True  # set gradient tape to True

            y_base, _ = nn_model(x_base)  # forward prop baseline
            y_pred, _ = nn_model(x_pred)  # forward prop

            y_base.sum().backward()  # baseline backpropagation
            y_pred.sum().backward()  # prediction backpropagation

            # Normalize the saliency plot
            img_source = x.squeeze().detach().numpy()
            img_baseline = torch.abs(x_base.grad.squeeze())
            img_saliency = torch.abs(x_pred.grad.squeeze())
            img_delta = x_pred.grad.squeeze() - x_base.grad.squeeze()
            img_overlay = img_saliency * img_source

            # Generate subplots
            column_idx = ii * n_cols + jj
            ax1 = plt.subplot(n_rows, n_cols, column_idx + 0 * n_cols)
            ax2 = plt.subplot(n_rows, n_cols, column_idx + 1 * n_cols)
            ax3 = plt.subplot(n_rows, n_cols, column_idx + 2 * n_cols)
            ax4 = plt.subplot(n_rows, n_cols, column_idx + 3 * n_cols)
            ax5 = plt.subplot(n_rows, n_cols, column_idx + 4 * n_cols)

            # plot images
            ax1.imshow(img_source, cmap=plt.cm.viridis, aspect="auto")
            ax2.imshow(img_baseline, cmap=plt.cm.viridis, aspect="auto")
            ax3.imshow(img_saliency, cmap=plt.cm.viridis, aspect="auto")
            ax4.imshow(img_delta, cmap=plt.cm.viridis, aspect="auto")
            ax5.imshow(img_overlay, cmap=plt.cm.viridis, aspect="auto")

            # Add labels for first column only
            if column_idx == 1:
                ax1.set_ylabel("source")
                ax2.set_ylabel("baseline")
                ax3.set_ylabel("saliency")
                ax4.set_ylabel("delta")
                ax5.set_ylabel("overlay")

            # Remove all ticks
            remove_ticks(ax1)
            remove_ticks(ax2)
            remove_ticks(ax3)
            remove_ticks(ax4)
            remove_ticks(ax5)

    # Save figure
    plt.savefig(f"{save_path}/{figure_name}_{global_step}.png", dpi=250)


from typing import Type


def remove_ticks(axis):
    axis.tick_params(
        axis="y",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        left=False,  # ticks along the bottom edge are off
        labelleft=False,  # labels along the left edge are off
    )

    axis.tick_params(
        axis="x",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )
