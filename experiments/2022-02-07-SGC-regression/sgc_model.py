from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Union
from torch_geometric.nn import SGConv
import mlflow
from mlflow import log_params, set_tags, log_artifact
from deepspparks.visualize import regression_results_plot


class SGCNet(torch.nn.Module):
    def __init__(
        self,
        k: int,
        data: Union[Data, None] = None,
        num_features: Union[int, None] = None,
        log: bool = True,
    ):
        super().__init__()

        # get number of features (input shape) and number of classes (output shape)
        if data is not None:
            num_features = data.num_features

        # SGCNet only has one layer, the SGConv layer
        # multiple iterations of message passing handled with K parameter
        # from pyg documentation: cached should only be set to true for transductive
        # learning (ie where the same graph is used and only the labels of some nodes
        # are unknown)
        self.conv = SGConv(in_channels=num_features, out_channels=1, K=k, cached=False)

        # forces weight tensors to double type, preventing subtle errors later
        self.double()

        # for logging
        self.model_name = "SGCNet-regression-v1"
        self.k = k
        if log:
            self._log()

    def forward(self, data):
        x = self.conv(data.x, data.edge_index)
        # growth ratio must be non-negative
        # thus, we square the output to prevent nn from
        # predicting invalid values
        return torch.square(x)

    # predict needed for mlflow.torch.save_model to include pyfunc
    def predict(self, data, mask=None):
        yp = self.forward(data)
        if mask is None:
            return yp
        else:
            return yp[mask]

    def _log(self):
        set_tags({"model": self.model_name})
        log_params(
            {
                "k": self.k,
            }
        )

    def mlflow_save_state(self, path):
        torch.save(self.state_dict(), path)
        log_artifact(path)


def train_step(model, batch, optimizer, mask):
    """
    Single step of training loop.

    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    batch: torch_geometric.data.Data object
        data to train on

    optimizer: torch.optim.Optimizer object
        optimizer for model

    mask: torch.Tensor
        [N-node] element boolean tensor.
        If mask[n] == True, loss from prediction on node n is used to update gradients
        during training. Otherwise, the prediction is ignored during training.

    Returns
    -------
    None

    Other
    -------
    Updates model parameters in-place.
    """

    model.train()  # set training mode to True (enable dropout, etc)

    optimizer.zero_grad()  # reset gradients

    yp = model(batch)  # call forward on model
    yt = batch.y.double().unsqueeze(1)  # ground truth y values, as nx1 tensor

    # torch.nn.functional.nll_loss(input, target)
    # input: NxC double tensor. input[i,j] contains the log-probability (ie log-softmax)
    #                             prediction for class j of sample i
    # target: N-element int tensor N[i] gives the ground truth class label for sample i
    # forward() needs all nodes for message passing
    # however, when computing loss, we only consider nodes included in mask to avoid
    # including unwanted nodes (ie validation/test nodes, non-candidate grains, etc)
    mse_loss = F.mse_loss(yp[mask], yt[mask])
    mse_loss.backward()  # compute gradients
    optimizer.step()  # update model params

    return


def train_loop(
    model,
    data_train,
    data_valid,
    data_test,
    optimizer,
    max_iterations,
    checkpoint_iterations,
    artifact_root,
):
    """
    Training a single model with a fixed set of parameters.

    Logs various parameters to mlflow tracking server


    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    data_train, data_valid, data_test: torch_geometric.data.Data
        training, validation, testing data to train/evaluate model on
        test data only used for final model

    optimizer: torch.optim.Optimizer object
        optimizer for model

    max_iterations: int

    checkpoint_iterations

    artifact_root: Path object
        temp path for local artifact storage BEFORE calling mlflow.log_artifact()

    Returns
    --------
    best_model: OrderedDict
        state dict (model.state_dict()) from model with lowest validation loss

    """
    print("Training model")
    # store initial params and initial accuracy/loss for nicer training graphs
    te, tl = mean_abs_error_and_mse_loss(
        model, data_train, data_train.candidate_mask
    )  # train accuracy, loss
    ve, vl = mean_abs_error_and_mse_loss(
        model, data_valid, data_valid.candidate_mask
    )  # val accuracy, loss

    best_params = model.state_dict()
    artifact_path = Path(
        artifact_root, "model_state_dict_checkpoint_{:04d}.pt".format(0)
    )
    torch.save(best_params, artifact_path)
    mlflow.log_artifact(str(artifact_path.absolute()), "model_checkpoints")
    mlflow.log_metrics(
        {
            "fit_train_error": te,
            "fit_train_loss": tl,
            "fit_val_error": ve,
            "fit_val_loss": vl,
        },
        step=0,
    )
    best_val_loss = 1e10
    best_iter = 0
    msg = (
        "  iteration {:>4d}/{:<4d} Train abs error: {:.4f}, Val abs error: {:.4f},"
        " Train loss: {:.4f}, Val loss: {:.4f}"
    )

    # train model. Store train/val acc/loss at desired checkpoint iterations
    iters = range(checkpoint_iterations, max_iterations + 1, checkpoint_iterations)
    t = tqdm(iters, desc=msg.format(0, max_iterations, te, ve, tl, vl), leave=True)
    for train_iter in t:
        for _ in range(
            checkpoint_iterations
        ):  # train for number of iterations in each checkpoint period
            train_step(model, data_train, optimizer, data_train.candidate_mask)

        # at the end of the checkpoint period, record loss and accuracy metrics
        te, tl = mean_abs_error_and_mse_loss(
            model, data_train, data_train.candidate_mask
        )  # train error, loss
        ve, vl = mean_abs_error_and_mse_loss(
            model, data_valid, data_valid.candidate_mask
        )  # val error, loss
        savepath = Path(
            artifact_root, "model_state_dict_checkpoint_{:04d}.pt".format(train_iter)
        )
        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics(
            {
                "fit_train_error": te,
                "fit_train_loss": tl,
                "fit_val_error": ve,
                "fit_val_loss": vl,
            },
            step=train_iter,
        )

        # update progress bar
        t.set_description(msg.format(train_iter, max_iterations, te, ve, tl, vl))
        t.refresh()

        # if val loss is lower than previous best, update best val loss and params
        if vl < best_val_loss:
            best_iter = train_iter
            best_val_loss = vl
            best_train_loss = tl
            best_train_error = te
            best_val_error = ve
            best_params = model.state_dict()

    print("\nlogging model")
    # logging best model
    model.load_state_dict(best_params)
    # get gt and yp values for train, val, test, and plot them
    ypt = model.predict(data_train)[data_train.candidate_mask].detach().numpy()
    ypv = model.predict(data_valid)[data_valid.candidate_mask].detach().numpy()
    gtt = data_train.y[data_train.candidate_mask].detach().numpy()
    gtv = data_valid.y[data_valid.candidate_mask].detach().numpy()
    yptest = model.predict(data_test)[data_test.candidate_mask].detach().numpy()
    gttest = data_test.y[data_test.candidate_mask].detach().numpy()
    figpath = artifact_root / "train_results.html"
    fig = regression_results_plot(
        gtt, ypt.squeeze(), gtv, ypv.squeeze(), gttest, yptest.squeeze(), "cgr values"
    )
    fig.write_html(figpath)
    log_artifact(str(figpath), "figures")

    test_error, test_loss = mean_abs_error_and_mse_loss(
        model, data_test, data_test.candidate_mask
    )

    best_metrics = {
        "best_iter": best_iter,
        "train_error": best_train_error,
        "train_loss": best_train_loss,
        "val_error": best_val_error,
        "val_loss": best_val_loss,
        "test_error": test_error,
        "test_loss": test_loss,
    }
    mlflow.log_metrics(best_metrics)  # log iteration with lowest val loss
    return best_params, best_metrics


@torch.no_grad()  # speed up predictions by disabling gradients
def mean_abs_error_and_mse_loss(model, data, mask):
    """
        Compute mea

        Parameters
        ----------
    ----------
        model: torch.nn.Module object
            model to train

        data: torch_geometric.data.Data object
            data to evaluate model on
        y: Tensor
            targets of ground truth values for each item in data


        mask: torch.Tensor
            [N-node] element boolean tensor.
            If mask[n] == True, loss from prediction on node n is included in accuracy
            and loss calculations.
            Otherwise, the prediction for node n is ignored.

        Returns
        -------
        acc, loss: float (may be 0-dim torch tensor)
            mean accuracy and nll-loss of predictions
    """
    model.train(False)

    yp = model(data)[mask]

    yt = data.y[mask].double().unsqueeze(1)  # ground truth values as nx1 tensor

    loss = F.mse_loss(yp, yt)  #
    mean_abs_error = torch.mean(torch.abs(yt - yp))

    return float(mean_abs_error), float(loss)
