from torch.nn import Module
from torch.nn.functional import mse_loss
from data import Dataset
from deepspparks.paths import ARTIFACT_PATH
from deepspparks.visualize import agg_cm
from pathlib import Path
from torch_geometric.loader import DataLoader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Union, Dict
from numbers import Number
from copy import deepcopy


def train_step(model: Module, batch, optimizer):
    model.train()  # set model to training mode (enable gradients, dropout, etc)
    optimizer.zero_grad()  # reset gradient
    y_pred = model(batch)  # generate predictions
    # for autoencoder-decoder, y_true is just x
    loss = mse_loss(y_pred, batch.y)  # compute loss
    loss.backward()  # compute gradients with backprop
    optimizer.step()  # update weights


def train_loop(
    model: Module,
    data: Dataset,
    lr: float,
    decay: float,
    params: dict,
    artifact_root: Union[str, Path] = ARTIFACT_PATH,
) -> Union[dict, Dict[str, Number]]:
    model = model.float().to(data.device)
    train_loader = DataLoader(
        dataset=data.train,
        batch_size=params["loader"]["batch_size"]["train"],
        shuffle=True,
        num_workers=params["loader"]["num_workers"],
    )
    val_loader = DataLoader(
        dataset=data.val,
        batch_size=params["loader"]["batch_size"]["val"],
        shuffle=False,
        num_workers=params["loader"]["num_workers"],
    )

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=decay,
    )

    scheduler = ReduceLROnPlateau(optimizer)

    mlflow.log_params(
        {
            "train_samples_per_epoch": len(data.train),
            "batch_size": params["loader"]["batch_size"]["train"],
            "lr": lr,
            "decay": decay,
        }
    )

    tl = batch_mse_loss(model, train_loader, data.device)
    vl = batch_mse_loss(model, val_loader, data.device)

    mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=0)

    best_metrics = {
        "train_loss": tl,
        "val_loss": vl,
        "best_epoch": 0,
    }
    best_weights = deepcopy(model.state_dict())

    max_epoch = params["training"]["max_epoch"]
    checkpoint_epoch = params["training"]["checkpoint_epoch"]

    epochs = range(checkpoint_epoch, max_epoch + 1, checkpoint_epoch)
    msg = "epoch {:>3d}/{:<3d} Train loss: {:.4e}, Val loss: {:.4e}"
    t = tqdm(epochs, desc=msg.format(0, max_epoch, tl, vl, leave=True))

    for train_epoch in t:
        for _ in range(checkpoint_epoch):  # train for n epochs before checkpointing
            for batch in train_loader:
                # aggregate final states from repeated growth trials into single
                # value

                batch = batch.to(device=data.device)
                train_step(model, batch, optimizer)

        # checkpointing
        tl = batch_mse_loss(model, train_loader, data.device)
        vl = batch_mse_loss(model, val_loader, data.device)

        # keep track of best performing checkpoint
        if vl < best_metrics["val_loss"]:
            best_metrics = {
                "train_loss": tl,
                "val_loss": vl,
                "best_epoch": train_epoch,
            }
            best_weights = deepcopy(model.state_dict())

        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:03d}".format(train_epoch)
        )

        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics(
            {"fit_train_loss": tl, "fit_val_loss": vl},
            step=train_epoch,
        )

        # step down the learning rate, if applicable
        scheduler.step(vl)

        # update progress bar
        t.set_description(msg.format(train_epoch, max_epoch, tl, vl))
        t.refresh()

    print("logging model")
    model.load_state_dict(best_weights)
    model.eval()

    test_loader = DataLoader(
        dataset=data.test,
        batch_size=params["loader"]["batch_size"]["test"],
        shuffle=False,
        num_workers=params["loader"]["num_workers"],
    )

    # train/vall acc/loss already stored in best_metrics
    # but we still need the confusion matrices for final evaluation
    # (requires detaching/numpy conversion so this is not done at every step of
    #  train loop)
    _, _, train_cm = batch_mse_loss(
        model, train_loader, data.device, return_cm=True
    )

    _, _, val_cm = batch_mse_loss(
        model, val_loader, data.device, return_cm=True
    )

    test_acc, test_loss, test_cm = batch_mse_loss(
        model, test_loader, data.device, return_cm=True
    )

    agg_cm(
        [train_cm, val_cm, test_cm],
        return_figure=False,
        fpath=ARTIFACT_PATH / "confusion_matrix.png",
        artifact_path="figures/",
    )

    for k, v in zip(("test_loss", "test_acc"), (test_loss, test_acc)):
        best_metrics[k] = v

    mlflow.log_metrics(best_metrics)
    mlflow.pytorch.log_model(model, "best_model")

    return best_weights, best_metrics


def batch_mse_loss(
    model: Module, dataloader: DataLoader, device: str = "cpu", return_cm: bool = False
):
    model = model.to(device=device, dtype=float)
    model.eval()
    mean_loss = 0.0
    mean_acc = 0.0
    total_samples = 0
    if return_cm:
        cm = np.zeros((2, 2), dtype=float)
    for batch in dataloader:
        batch.y = batch.y.to(device=device, dtype=torch.long)
        batch.x = batch.x.to(device=device, dtype=float)
        yp = model(batch)
        # average loss per sample in batch
        loss = float(mse_loss(yp, batch.y, reduction="mean").detach())

        return loss
