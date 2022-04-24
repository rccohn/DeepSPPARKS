from torch.nn import Module
from torch.nn.functional import nll_loss
from data import Dataset
from typing import Union
from deepspparks.paths import ARTIFACT_PATH
from pathlib import Path
from torch_geometric.data import DataLoader
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from tqdm import tqdm


def train_step(model: Module, batch, optimizer):
    model.train()  # set model to training mode (enable gradients, dropout, etc)
    optimizer.zero_grad()  # reset gradient
    y_pred = model(batch)  # generate predictions
    # for autoencoder-decoder, y_true is just x
    loss = nll_loss(batch, y_pred)  # compute loss
    loss.backward()  # compute gradients with backprop
    optimizer.step()  # update weights


def train_loop(
    model: Module,
    data: Dataset,
    lr,
    decay,
    params: dict,
    artifact_root: Union[str, Path] = ARTIFACT_PATH,
) -> Module:
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
        }
    )

    tl = batch_nll_loss(model, train_loader, data.device)
    vl = batch_nll_loss(model, val_loader, data.device)

    mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=0)

    best_val_loss = vl

    max_epoch = params["training"]["max_epoch"]
    checkpoint_epoch = params["training"]["checkpoint_epoch"]

    epochs = range(checkpoint_epoch, max_epoch + 1, checkpoint_epoch)
    msg = "epoch {:>3d}/{:<3d} Train loss: {:.4e}, Val loss: {:.4e}"
    t = tqdm(epochs, desc=msg.format(0, max_epoch, tl, vl, leave=True))

    for train_epoch in t:
        for _ in range(checkpoint_epoch):  # train for n epochs before checkpointing
            for batch in train_loader:
                batch = batch.float().to(data.device)
                train_step(model, batch, scheduler)

        # checkpointing
        tl = batch_nll_loss(model, train_loader, data.device)
        vl = batch_nll_loss(model, val_loader, data.device)
        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:03d}".format(train_epoch)
        )

        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=train_epoch)

        # update progress bar
        t.set_description(msg.format(train_epoch, max_epoch, tl, vl))
        t.refresh()

        # keep track of best performing checkpoint
        if vl < best_val_loss:
            best_epoch = train_epoch
            best_val_loss = vl
            best_train_loss = tl
            best_params = model.state_dict()

    print("logging model")
    model.load_state_dict(best_params)
    model.eval()

    test_loader = DataLoader(
        dataset=data.test,
        batch_size=params["loader"]["batch_size"]["test"],
        shuffle=False,
        num_workers=params["loader"]["num_workers"],
    )

    test_loss = batch_nll_loss(model, test_loader, data.device)

    best_metrics = {
        "best_epoch": best_epoch,
        "train_loss": best_train_loss,
        "val_loss": best_val_loss,
        "test_loss": test_loss,
    }

    mlflow.log_metrics(best_metrics)
    mlflow.pytorch.log_model(model, "best_model")

    return model


def batch_nll_loss(model: Module, dataloader: DataLoader, device: str = "cpu"):
    model = model.to(torch.float).to(device)
    model.eval()
    mean_loss = 0.0
    total_samples = 0
    for batch, labels in dataloader:
        batch = batch.float().to(device)
        labels = labels.long().to(device)
        yp = model(batch)
        loss = float(nll_loss(yp, labels, reduction="mean").detach())
        m = len(labels)
        mean_loss += (loss - (m * mean_loss)) / (total_samples + m)
        total_samples += m

    return mean_loss
