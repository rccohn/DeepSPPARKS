from torch.nn import Module
from torch.nn.functional import nll_loss
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


def train_step(model: Module, batch, optimizer):
    model.train()  # set model to training mode (enable gradients, dropout, etc)
    optimizer.zero_grad()  # reset gradient
    mask = batch.grain_types == 0
    y_pred = model(batch)  # generate predictions
    # for autoencoder-decoder, y_true is just x
    loss = nll_loss(y_pred[mask], batch.y[mask])  # compute loss
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

    ta, tl = batch_acc_and_nll_loss(model, train_loader, data.device)
    va, vl = batch_acc_and_nll_loss(model, val_loader, data.device)

    mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=0)

    best_metrics = {
        "train_acc": ta,
        "train_loss": tl,
        "val_acc": va,
        "val_loss": vl,
        "best_epoch": 0,
    }
    best_weights = model.state_dict()

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
        ta, tl = batch_acc_and_nll_loss(model, train_loader, data.device)
        va, vl = batch_acc_and_nll_loss(model, val_loader, data.device)
        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:03d}".format(train_epoch)
        )

        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics(
            {"fit_train_loss": tl, "fit_val_loss": vl},
            step=train_epoch + checkpoint_epoch,
        )

        scheduler.step(vl)

        # update progress bar
        t.set_description(msg.format(train_epoch, max_epoch, tl, vl))
        t.refresh()

        # keep track of best performing checkpoint
        if vl < best_metrics["val_loss"]:
            best_metrics = {
                "train_acc": ta,
                "train_loss": tl,
                "val_acc": va,
                "val_loss": vl,
                "best_epoch": train_epoch + checkpoint_epoch,
            }
            best_weights = model.state_dict()

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
    _, _, train_cm = batch_acc_and_nll_loss(
        model, train_loader, data.device, return_cm=True
    )

    _, _, val_cm = batch_acc_and_nll_loss(
        model, val_loader, data.device, return_cm=True
    )

    test_acc, test_loss, test_cm = batch_acc_and_nll_loss(
        model, test_loader, data.device, return_cm=True
    )

    agg_cm(
        [train_cm, val_cm, test_cm],
        return_figure=False,
        fpath=ARTIFACT_PATH / "confusion_matrix.png",
        artifact_path="figures/",
    )

    # python3.8 doesn't have dict union yet,
    # so we append test_accuracy and test_loss using a dict comprehension
    best_metrics = {
        k: v
        for x in [best_metrics, {"test_acc": test_acc, "test_loss": test_loss}]
        for k, v in x.items()
    }

    mlflow.log_metrics(best_metrics)
    mlflow.pytorch.log_model(model, "best_model")

    return best_weights, best_metrics


def batch_acc_and_nll_loss(
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
        mask = batch.grain_types == 0
        batch.y = batch.y[mask].to(device=device, dtype=torch.long)
        batch.x = batch.x.to(device=device, dtype=float)
        log_probs = model(batch)[mask]
        # average loss per sample in batch
        loss = float(nll_loss(log_probs, batch.y, reduction="mean").detach())

        yp = log_probs.argmax(1)
        m = len(batch.y)

        # accuracy of predictions in batch
        acc = float((yp == batch.y).sum().detach()) / m

        if return_cm:
            cm += confusion_matrix(
                batch.y.detach().cpu().numpy(), yp.detach().cpu().numpy(), labels=[0, 1]
            )

        denom = (total_samples / m) + 1  # denominator of running mean expression,
        # (same for both acc and loss)

        mean_loss += (loss - mean_loss) / denom
        mean_acc += (acc - mean_acc) / denom
        total_samples += m

    if return_cm:
        return mean_acc, mean_loss, cm
    return mean_acc, mean_loss
