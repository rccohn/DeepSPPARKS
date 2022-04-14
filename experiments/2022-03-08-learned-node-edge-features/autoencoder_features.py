from pathlib import Path
from data import Dataset
import os
from shutil import rmtree
from deepspparks.utils import load_params
from deepspparks.visualize import scree_plot
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from torch.utils.data import DataLoader
from models import batch_mse_loss
import pandas as pd
import plotly.express as px
from nn_train import train_loop
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torch.nn.functional import mse_loss


def pca_train(params: dict, dataloader):
    """
    Parameters
    ----------
    params: dict
        loaded param file

    datasets: Dataset object
        contains data used in experiment
    """
    # create datasets and incremental PCA model
    # incremental PCA only needs 1 epoch to fit, but each batch must be at least
    # min(num_samples, num_features) to compute updated covariance matrix

    # after fitting model with all components, run inference with increasing number
    # of components. similar to variance vs n_components, compute loss vs n_components
    from sklearn.decomposition import IncrementalPCA

    # fit incremental pca to data batches

    # might need to set batch size in params,
    # min(n_components, n_samples) might be huge for patches
    for batch in dataloader:
        break
    n_components = params["pca"]["n_components"]
    model = IncrementalPCA(n_components=n_components)
    # todo log n_components
    for batch in dataloader:
        batch = batch.detach().numpy().reshape(len(batch), -1)

        model.partial_fit(batch)

    # record variance preserved per number of components
    fpath = "/root/artifacts/pca_explained_var.npy"
    assert Path(fpath).parent.exists()
    np.save(fpath, model.explained_variance_, allow_pickle=False)

    mlflow.log_artifact(fpath, artifact_path="pca/")
    figpath = "/root/artifacts/pca_scree.html"
    fig = scree_plot(model.explained_variance_ratio_)
    fig.write_html(figpath)
    mlflow.log_artifact(figpath, artifact_path="pca/")

    # save pca model
    mlflow.sklearn.log_model(model, artifact_path="pca/pca_model")
    return model


def pca_evaluate(dataset, model, params):
    # dataset contains train, val, and test subsets
    from models import PCAEncoder

    # wrap pca model into one that can take
    # in 3d collections of 2d torch tensors
    # instead of a 2d data numpy array
    model = PCAEncoder(model)
    # measure performance with several fractions of variance
    var = model.pca_model.explained_variance_ratio_.cumsum()
    n_components = [np.argmax(var >= x) + 1 for x in (0.3, 0.5, 0.8, 0.9, 0.95, 0.96)]
    dfs = []
    for title, subset in zip(
        ("train", "val", "test"), (dataset.train, dataset.val, dataset.test)
    ):

        loader = DataLoader(subset, batch_size=2, shuffle=False)

        losses = np.zeros(len(n_components), dtype=float)
        vars = np.zeros_like(losses)
        for i, n in enumerate(n_components):
            model.n_components = n
            for tv, data in zip(("train", "val"), (dataset.train, dataset.val)):
                for idx in params["sample_results"][tv]:
                    sample_performance(model, tv, data, idx, pca=True)

            losses[i] = float(batch_mse_loss(model, loader))
            vars[i] = var[n + 1]
        df_sub = pd.DataFrame(
            {"n_components": n_components, "var": vars, "mse_loss": losses.tolist()}
        )
        df_sub["dataset"] = title
        dfs.append(df_sub)

    df = pd.concat(dfs, ignore_index=True)
    savepath = "/root/artifacts/results_df.csv"
    df.to_csv(savepath)
    mlflow.log_artifact(savepath, artifact_path="results/")
    savepath = "/root/artifacts/results_df.html"
    df.to_html(savepath)
    mlflow.log_artifact(savepath, artifact_path="results/")

    figpath = "/root/artifacts/loss_vs_nc.html"
    fig = px.line(
        data_frame=df,
        x="n_components",
        y="mse_loss",
        line_group="dataset",
        color="dataset",
        markers=True,
    )
    fig.update_layout(hovermode="x unified", font={"size": 14})
    fig.write_html(figpath)
    mlflow.log_artifact(figpath, artifact_path="results/")
    figpath = "/root/artifacts/loss_vs_var.html"
    fig = px.line(
        data_frame=df,
        x="var",
        y="mse_loss",
        line_group="dataset",
        color="dataset",
        markers=True,
    )
    fig.update_layout(hovermode="x unified", font={"size": 14})
    fig.write_html(figpath)
    mlflow.log_artifact(figpath, artifact_path="results/")


def autoencoder_train_and_evaluate(params: dict, dataset: Dataset):
    from models import get_autoencoder

    model = get_autoencoder(params["autoencoder"]["architecture"])()
    model = train_loop(model, dataset, params)
    return model
    # goal is to create a neural autoencoder to achive lower loss than pca with same
    # number of components, or same loss compared to pca model with fewer components
    # params should allow for standard training parameters (learning rate, weight decay,
    # epochs, etc.)
    # It might be difficult to parameterize model architectures since there are so many,
    # so you might just have to create multiple different models and see what works.

    # it might be good to first start with pca to get an idea of how many components
    # to shoot for, (look at reconstructions to see qualitative quality as well
    # as variance/loss) and then use this as a guideline for the autoencoder model.


def main():
    param_file = "/root/inputs/params.yaml"
    params = load_params(param_file)
    name = params["mlflow"]["dataset_name"]
    device = None if params["which"]["model"] != "pca" else "cpu"
    dataset = Dataset(
        name,
        params["which"]["patches"],  # indicates node vs edge patches
        device=device,
    )
    with mlflow.start_run(nested=False):
        mlflow.log_artifact(param_file)
        mlflow.log_params(
            {
                "patches": params["which"]["patches"],
                "pca_or_autoencoder": params["which"]["model"],
            }
        )

        dataset.process()

        if params["which"]["model"] == "pca":
            dataloader = DataLoader(
                dataset.train,
                batch_size=params["dataloader"]["batch_size"],
                drop_last=True,
            )

            model = pca_train(params, dataloader)
            pca_evaluate(dataset, model, params)
        else:
            model = autoencoder_train_and_evaluate(params, dataset)

            # only generate sample images for autoencoder
            # pca is ambiguous as results depend on number of components
            # manual comparison can easily be done later
            for tv, data in zip(("train", "val"), (dataset.train, dataset.val)):
                for idx in params["sample_results"][tv]:
                    sample_performance(model, tv, data, idx)


def sample_performance(model, tv, data, idx, pca=False):
    # if pca -> number of components is stored

    model = model.to(data.device)

    # generate figures
    batch = data[idx].unsqueeze(0).to(data.device)
    pred = model.predict(batch).detach()
    loss = float(mse_loss(batch, pred).detach().cpu().numpy())

    cscale = "plotly3"
    fig = make_subplots(
        1,
        3,
        shared_yaxes=True,
        subplot_titles=["Original", "Reconstructed", "Original - reconstructed"],
    )
    img_gt = batch.detach().cpu().numpy().squeeze()
    img_pred = pred.squeeze().cpu().numpy()
    diff = img_gt - img_pred
    zmin = min((img_gt.min(), img_pred.min(), diff.min()))
    zmax = max((img_gt.max(), img_pred.max(), diff.max()))

    fig.add_trace(
        go.Heatmap(
            z=img_gt,
            showscale=False,
            colorscale=cscale,
            zmin=zmin,
            zmax=zmax,
        ),
        1,
        1,
    )
    fig.add_trace(
        go.Heatmap(
            z=img_pred,
            showscale=False,
            colorscale=cscale,
            zmin=zmin,
            zmax=zmax,
        ),
        1,
        2,
    )
    fig.add_trace(go.Heatmap(z=diff, zmin=zmin, zmax=zmax, colorscale=cscale), 1, 3)
    if pca:
        title_str = "{}-{}({} components): mse loss: {:.4e}".format(
            tv, idx, model.n_components, loss
        )
    else:
        title_str = "{}-{}: mse loss: {:.4e}".format(tv, idx, loss)
    fig.update_layout(title=title_str, font={"size": 14})

    # save figure and log as mlflow artifact under
    # run/sample_images/{train, val}/0...01.html
    figpath = Path("/root/artifacts/{}/{:09}.html".format(tv, idx))
    if not figpath.parent.exists():
        os.makedirs(figpath.parent, exist_ok=True)
    fig.write_html(str(figpath))
    if pca:
        artifact_path = "sample_images/{}-components/{}".format(model.n_components, tv)
    else:
        artifact_path = "sample_images/{}".format(tv)

    mlflow.log_artifact(str(figpath), artifact_path=artifact_path)

    # auto-encoder is a very open ended problem
    # the main things we want to do are:
    #  1) generate a reasonable pca baseline: generate some representative images of
    #     reconstruction, and measure MSE loss as a function the number of components
    #  2) evaluate one or more autoencoder architectures to convert 2d images into
    #     compressed 1d vectors, and compare the loss to PCA with the same number of
    #     components
    # we want to do this for both nodes and edges
    # note that for candidate grain simulations, patches are very, very simple
    # with only 3/5 intensity values for node/edge patches, and consist of very simple
    # filled-in shapes. It is very possible that PCA is sufficient and will therefore
    # be difficult to beat. Thus, part of the goal is setting up future studies with
    # more realistic systems. If we can get a neural autoencoder to work, then it is
    # much likely to generalize better for more complicated systems.

    # TODO log which encoder (pca vs autoencoder), which patches (nodes vs edges)
    # TODO for both pca and autoencoder, log examples of predictions on select train/val
    #     images (can be parameterized). imshow original, reconstruction, diff
    #     and title with loss


def test():
    # verify that data loader correctly works and patches are correctly displayed
    param_file = "/root/inputs/params.yaml"
    params = load_params(param_file)
    name = params["mlflow"]["dataset_name"]

    out_root = Path("/root/data/processed/{}".format(name))
    for which in ("node", "edge"):
        ds = Dataset(name=name, which=which)
        ds.process()
        for sub in ("train", "val", "test"):
            sub_root = out_root / which / sub
            if sub_root.exists():
                rmtree(sub_root)
            os.makedirs(sub_root, exist_ok=True)
            loader = ds.__getattribute__("{}_loader".format(sub))
            i = 0
            for img in loader:
                if i == 3:
                    break
                img = img[0].detach().numpy()
                fig, ax = plt.subplots(dpi=150)
                ax.imshow(img, vmin=0.2, vmax=0.9, cmap="magma")
                fig.tight_layout()
                fig.savefig(sub_root / "{}.png".format(i), bbox_inches="tight")
                i += 1


if __name__ == "__main__":
    main()
