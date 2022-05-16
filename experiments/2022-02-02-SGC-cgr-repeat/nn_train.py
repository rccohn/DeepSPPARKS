from deepspparks.paths import ARTIFACT_PATH
import torch
from torch.nn.functional import nll_loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from pathlib import Path
from deepspparks.visualize import agg_cm


def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()
    mask = batch.candidate_mask
    y_pred = model(batch)

    loss = nll_loss(y_pred[mask], batch.y[mask])
    loss.backward()
    optimizer.step()


def train_loop(
    model,
    data_train,
    data_val,
    data_test,
    device,
    lr,
    decay,
    params,
    artifact_root=ARTIFACT_PATH,
):
    model = model.float().to(device)
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=decay,
    )

    scheduler = ReduceLROnPlateau(optimizer)

    mlflow.log_params(
        {
            "learning_rate": lr,
            "weight_decay": decay,
            "loss": "nll",
            "optimizer": "Adam",
        }
    )

    ta, tl = batch_acc_and_nll_loss(model, data_train)
    va, vl = batch_acc_and_nll_loss(model, data_val)

    mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=0)

    best_metrics = {
        "train_acc": ta,
        "train_loss": tl,
        "val_acc": va,
        "val_loss": vl,
        "best_epoch": 0,
    }

    best_weights = model.state_dict()

    max_epoch = params["training"]["max_iter"]
    checkpoint_epoch = params["training"]["checkpoint_iter"]

    epochs = range(checkpoint_epoch, max_epoch + 1, checkpoint_epoch)
    msg = "epoch {:>3d}/{:<3d} Train loss: {:.4e}, Val loss: {:.4e}"
    t = tqdm(epochs, desc=msg.format(0, max_epoch, tl, vl, leave=True))

    for train_epoch in t:
        for _ in range(checkpoint_epoch):
            train_step(model, data_train, optimizer)

        # checkpointing
        ta, tl = batch_acc_and_nll_loss(model, data_train, device)
        va, vl = batch_acc_and_nll_loss(model, data_val, device)

        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:03d}.pt".format(train_epoch)
        )
        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(savepath, "model_checkpoints")
        mlflow.log_metrics({"fit_train_loss": tl, "fit_val_loss": vl}, step=train_epoch)

        scheduler.step(vl)

        # update progress bar
        t.set_description(msg.format(train_epoch, max_epoch, tl, vl))
        t.refresh()

        # track best performing checkpoint
        if vl < best_metrics["val_loss"]:
            best_metrics = {
                "train_acc": ta,
                "train_loss": tl,
                "val_acc": va,
                "val_loss": vl,
                "best_epoch": train_epoch,
            }
            best_weights = model.state_dict()

    # log final model
    model.load_state_dict(best_weights)
    model.eval()

    _, _, train_cm = batch_acc_and_nll_loss(model, data_train, device, True)
    _, _, val_cm = batch_acc_and_nll_loss(model, data_val, device, True)

    test_acc, test_loss, test_cm = batch_acc_and_nll_loss(
        model, data_test, device, True
    )

    agg_cm(
        [train_cm, val_cm, test_cm],
        return_figure=False,
        fpath=ARTIFACT_PATH / "confusion_matrix.png",
        artifact_path="figures/",
    )

    best_metrics["test_acc"] = test_acc
    best_metrics["test_loss"] = test_loss

    mlflow.log_metrics(best_metrics)

    return best_weights, best_metrics


def batch_acc_and_nll_loss(model, data, device="cpu", return_cm=False):
    model = model.to(device=device, dtype=torch.float)
    model.eval()
    mask = data.candidate_mask
    y = data.y[mask].to(device=device, dtype=torch.long)
    data.x = data.x.to(device=device, dtype=torch.float)
    log_probs = model(data)[mask]
    loss = float(nll_loss(log_probs, y, reduction="mean").detach())
    yp = log_probs.argmax(1)
    acc = float((yp == y).sum().detach() / len(y))

    if return_cm:
        cm = confusion_matrix(
            y.detach().cpu().numpy(), yp.detach().cpu().numpy(), labels=[0, 1]
        )
        return acc, loss, cm
    return acc, loss
