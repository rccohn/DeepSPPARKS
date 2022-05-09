#  standard training for  neural autoencoders
from torch.optim import Adam, Optimizer
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss
from data import Dataset
from typing import Union, Tuple
from pathlib import Path
from models import batch_mse_loss
import mlflow
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from deepspparks.paths import ARTIFACT_PATH


def train_step(model: Module, batch: Tensor, optimizer: Optimizer) -> None:
    """
    Single training step for neural auto-encoder.

    """
    model.train()  # set model to training mode
    optimizer.zero_grad()  # reset gradient
    y_pred = model(batch)  # generate predictions
    # for autoencoder-decoder, y_true is just x
    loss = mse_loss(batch, y_pred)  # compute loss
    loss.backward()  # compute gradients with backprop
    optimizer.step()  # update weights


def train_loop(
    model: Module,
    data: Dataset,
    params: dict,
    artifact_root: Union[str, Path] = ARTIFACT_PATH,
) -> Tuple[dict, dict]:
    """
    todo docstring
    """
    # move model to same device that data will be loaded to
    model = model.float().to(data.device)

    # create loaders for training and validation set, and optimizer for training model
    train_loader = DataLoader(
        data.train,
        batch_size=params["dataloader"]["batch_size"],
        shuffle=True,  # possible torch issue using shuffle=True
        num_workers=params["dataloader"]["num_workers"],
        drop_last=False,
    )
    val_loader = DataLoader(
        data.val,
        batch_size=params["dataloader"]["batch_size"],
        num_workers=params["dataloader"]["num_workers"],
        drop_last=False,
    )

    optimizer = Adam(
        model.parameters(),
        lr=params["optimizer"]["lr"],
        weight_decay=params["optimizer"]["decay"],
    )

    # allow epochs to be translated to number of samples/batches
    mlflow.log_params(
        {
            "samples_per_epoch": len(data.train),
            "batch_size": params["dataloader"]["batch_size"],
        }
    )

    # get loss values for initial random weights for prettier loss vs epoch graph
    tl = batch_mse_loss(model, train_loader, data.device)
    vl = batch_mse_loss(model, val_loader, data.device)
    mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=0)

    # during training, track the losses at each checkpoint so that the best can be
    # stored
    best_train_loss = tl
    best_val_loss = vl
    best_epoch = 0
    best_params = model.state_dict()

    # training params
    max_epoch = params["training"]["max_epoch"]
    checkpoint_epoch = params["training"]["checkpoint_epoch"]

    # info for checkpointing/displayign training progress
    epochs = range(checkpoint_epoch, max_epoch + 1, checkpoint_epoch)
    msg = "epoch {:>3d}/{:<3d} Train loss: {:.4f}, Val loss: {:.4f}"
    t = tqdm(epochs, desc=msg.format(0, max_epoch, tl, vl), leave=True)
    # train model, store weights and losses at desired checkpoints
    for train_epoch in t:  # continue until model trained for max epochs
        for _ in range(checkpoint_epoch):  # train for n epochs before checkpointing
            for batch in train_loader:  # each epoch, train on all batches in loader
                batch = batch.float().to(data.device)
                train_step(model, batch, optimizer)

        model.eval()
        # checkpoint: save model weighs, train loss, and val loss
        tl = batch_mse_loss(model, train_loader, data.device)
        vl = batch_mse_loss(model, val_loader, data.device)
        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:03d}".format(train_epoch)
        )

        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=train_epoch)

        # update progress bar
        t.set_description(msg.format(train_epoch, max_epoch, tl, vl))
        t.refresh()

        # if val_loss is lower than previous best, update best val loss and params
        if vl < best_val_loss:
            best_epoch = train_epoch
            best_val_loss = vl
            best_train_loss = tl
            best_params = model.state_dict()

    print("logging model")  # logging model after training is complete
    model.load_state_dict(best_params)  # best model = lowest validation loss
    model.eval()

    # compute performance on test set
    test_loader = DataLoader(
        data.test,
        batch_size=params["dataloader"]["batch_size"],
        num_workers=params["dataloader"]["num_workers"],
        drop_last=False,
    )
    test_loss = batch_mse_loss(model, test_loader, data.device)

    # save to tracking database
    best_metrics = {
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "test_loss": test_loss,
    }

    mlflow.log_metrics(best_metrics)
    # TODO for large models, maybe avoid redundant saving-> checkpoint, then
    #    "best" model, then mlflow model
    mlflow.pytorch.log_model(model, "best_model")

    return model
