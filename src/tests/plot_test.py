from os.path import exists

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from src.data import get_mnist_dataloaders
from src.model import BasicCNN
from src.plot import generate_projection
from src.plot import plot_confusion_matrix
from src.plot import saliency_map


def test_plot_confusion_matrix(tmpdir):
    """Generate a confusion matrix using a dataloader.  Used to exercise the code."""

    # Create a model and dataloader
    model = BasicCNN()
    _, dataloader_val, _, _ = get_mnist_dataloaders()
    figure_name = "confusion_matrix_test"

    # Exercise plotting
    save_path = f"{tmpdir}"
    plot_confusion_matrix(
        model=model,
        dataloader_val=dataloader_val,
        save_path=save_path,
        fig_name=figure_name,
    )

    # Check that a plot was generated
    assert exists(f"{save_path}/{figure_name}.png"), "Could not find confusion matrix image file!"


def test_generate_projection(tmpdir):
    """Generate a projection using an untrained model.  Used to exercise the code."""

    # Create a model, writer, and dataloader
    model = BasicCNN()
    log_path = str(tmpdir)
    writer = SummaryWriter(log_dir=log_path)
    _, dataloader_val, _, _ = get_mnist_dataloaders()

    # Exercise plotting
    generate_projection(model=model, dataloader_val=dataloader_val, writer=writer)

    # Verify that logs were created (does not check if they are correct!)
    assert exists(
        f"{log_path}/projector_config.pbtxt"
    ), "Could not find projector_config.pbtxt file!"


def test_saliency_map(tmpdir):
    """Generate a saliency map of the some sample images with the network's focus. This is using an untrained model so that focus areas will not be valid.  Used to exercise the code."""

    # Create a model
    model = BasicCNN()
    save_path = str(tmpdir)
    figure_name = "test"
    global_step = 12
    model_weights_path = f"{save_path}/model.pkl"
    torch.save(model.state_dict(), model_weights_path)

    # Get a Dataset
    _, _, _, dataset_val = get_mnist_dataloaders()

    saliency_map(
        model=model,
        dataset_val=dataset_val,
        figure_name=figure_name,
        global_step=global_step,
        model_weights_path=model_weights_path,
        save_path=save_path,
    )

    assert exists(
        f"{save_path}/{figure_name}_{global_step}.png"
    ), "Could not find saliency.png file!"
