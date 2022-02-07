import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import mlflow
import plotly
from plotly import graph_objects as go
import plotly.express as px


def scree_plot(pca_var):
    """
    Makes scree plot and returns plotly figure

    Parameters
    ----------
    pca_var: ndarray
        fraction of variance explained by each component
        (ie result of sklearn pca.explained_variance_ratio_)

    Returns
    -------
    fig: plotly figure
        figure with scree plot
    """
    components = list(range(1, len(pca_var) + 1))
    var_cumulative = pca_var.cumsum()
    df = make_plot_df(
        components,
        (pca_var, var_cumulative),
        hues=("individual", "cumulative"),
        x_label="n_components",
        y_label="fraction of variance explained",
    )
    fig = px.line(
        df,
        "n_components",
        "fraction of variance explained",
        color="hue",
        color_discrete_sequence=plotly.colors.qualitative.Dark2[2:4],
    )
    fig.update_layout(hovermode="x unified", font={"size": 14})

    return fig


def make_plot_df(xs, ys, hues=None, x_label="x", y_label="y", hue_label="hue"):
    """
    Convenience function to generate a dataframe ready for seaborn or plotly plotting.

    Generates a long-formatted dataframe with 3 columns. The first column contains x
    values, the second column contains y values, and the third column contains values
    that distinguish individual series. The returned dataframe df can then be plotted
    in seaborn or plotly by specifying a relevant plotting function with arguments like
    data=df, x=x_label, y=y_label, hue=hue_label

    Parameters
    ----------
    xs, ys: 1d or 2d list-like
        values to be included in dataframe and plotted on x or y axis.
        If either is 1d array, it will be broadcasted to match the dims of the other.
        If both are 1d arrays with length n, they will be broadcast into shape (1,n)

    hues: 1d array, numeric, string, or None
        if None, hues are not used.
        If specified, there should be 1 hue value for each series in the data.
        If numeric or string, the same hue will be given to all series.

    x_label, y_label: str
        column header for x and y values in dataframe

    hue_label: str or None
        column header for hue values. Can be None if hues is None, in which
        case dataframe will not include hue values.

    Returns
    -------
    df: pd.Dataframe
        Dataframe containing x, y, [hue (optional)] values
    """
    xs = np.asarray(xs).squeeze()
    if not xs.shape:
        xs = xs[np.newaxis]
    ys = np.asarray(ys).squeeze()
    if not ys.shape:
        ys = ys[np.newaxis]

    if hues is None:
        assert (xs.ndim == 1 or len(xs) == 1) and (
            ys.ndim == 1 or len(ys) == 1
        ), "xs and ys must be 1d if hues are not specified (hues is None)"
        return pd.DataFrame({x_label: xs.squeeze(), y_label: ys.squeeze()})

    # force xs and ys to be 2d
    if xs.ndim == 1:
        xs = xs[np.newaxis, :]
    if ys.ndim == 1:
        ys = ys[np.newaxis, :]

    assert (
        len(xs) == len(ys) or min(len(xs), len(ys)) == 1
    ), "xs and ys must be same shape or length 1 for broadcasting"
    # broadcast
    if len(xs) == 1:
        xs = np.broadcast_to(xs, ys.shape)
    elif len(ys) == 1:
        ys = np.broadcast_to(ys, xs.shape)

    # handle case where hues may be single value, or array of values.
    # if array, should be 1d-like (all but 1 dims should be size 1)
    hues = np.array(hues).squeeze()
    if not hues.shape:
        hues = hues[np.newaxis]

    if len(hues) == 1:
        hues = np.repeat(hues[0], len(xs))

    assert len(xs) == len(ys) and len(xs) == len(
        hues
    ), "hues must have 1 element per dataset or 1 element total"

    # put xs, ys, and hues in dataframe that can be conveniently plotted with
    # seaborn or plotly functions

    df = pd.concat(
        [
            pd.DataFrame({x_label: x, y_label: y, hue_label: hue})
            for x, y, hue in zip(xs, ys, hues)
        ]
    )
    return df


def pretty_cm(cm, labelnames, cscale=0.6, ax0=None, fs=6, cmap="cool"):
    """
    Generates a pretty-formated confusion matrix for convenient visualization.

    The true labels are displayed on the rows, and the predicted labels are displayed
    on the columns.

    Parameters
    ----------
    cm: ndarray
        nxn array containing the data of the confusion matrix.

    labelnames: list(string)
        list of class names in order on which they appear in the confusion matrix.
        For example, the first element should contain the class corresponding to the
        first row and column of *cm*.
    cscale: float
        parameter that adjusts the color intensity. Allows color to be present for
        confusion matrices with few mistakes, and controlling the intensity for ones
        with many misclassifications.

    ax0: None or matplotlib axis object
        if None, a new figure/axis will be created and displayed.
        if an axis is supplied, the confusion matrix will be plotted on it in place.
    fs: int
        font size for text on confusion matrix.

    cmap: str
        matplotlib colormap to use

    Returns
    ---------
    None

    """

    if ax0 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), dpi=300)
        fig.set_facecolor("w")
    else:
        ax = ax0

    n = len(labelnames)
    ax.imshow(np.power(cm, cscale), cmap=cmap, extent=(0, n, 0, n))
    labelticks = np.arange(n) + 0.5

    ax.set_xticks(labelticks, minor=True)
    ax.set_yticks(labelticks, minor=True)
    ax.set_xticklabels(["" for i in range(n)], minor=False, fontsize=fs)
    ax.set_yticklabels(["" for i in range(n)], minor=False, fontsize=fs)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels=labelnames, minor=True, fontsize=fs)
    ax.set_yticklabels(labels=reversed(labelnames), minor=True, fontsize=fs)

    ax.set_xlabel("Predicted Labels", fontsize=fs)
    ax.set_ylabel("Actual Labels", fontsize=fs)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(
            j + 0.5,
            n - i - 0.5,
            "{:^5}".format(z),
            ha="center",
            va="center",
            fontsize=fs,
            bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.3"),
        )
    ax.grid(which="major", color=np.ones(3) * 0.33, linewidth=1)

    if ax0 is None:
        ax.set_title("Accuracy: {:.3f}".format(cm.trace() / cm.sum()), fontsize=fs + 2)
        plt.show()
        return
    else:
        return ax


def agg_cm(cmlist, return_figure=True, fpath=None, artifact_path=None):
    """
    Display confusion matrix for train/validation/test performance and save to disk.

    Parameters
    ----------
    cmlist: list-like, or dict
        list-like structure containing confusion matrices in sklearn format
        (sklearn.metrics.confusion_matrix(y_true, y_predict)
         for the training, validation, and [optional test] sets respectively.
         if dict, keys should be ('train','val','test') and values should
         be the corresponding confusion matrix

    return_figure: Bool
        if True, figure is returned by function

    fpath: None or str or Path object
        if None, figure is not saved to disk or logged as artifact
        Otherwise, Path to save figure to.

    artifact_path: str or None
        if None, figure is not logged as artifact
        else:
         - fpath must not be none
         - if str, path in artifact directory to log artifact to with mlflow

    Returns
    ---------
    fig: matplotlib Figure or None
        if return_figure == True, figure is returned
        Otherwise, None is returned.

    Saves
    ---------
    cm_fig: image
        visualized confusion matrices saved to fname
    """
    if type(cmlist) == dict:
        cmlist = [cmlist["train"], cmlist["val"], cmlist.get("test", None)]
        cmlist = [
            x for x in cmlist if x is not None
        ]  # remove test if it is not in dict
    subsets = ["Train", "Valid", "Test"][: len(cmlist)]
    fig, ax = plt.subplots(
        1,
        len(cmlist),
        sharey=True,
        dpi=300,
        facecolor="w",
        figsize=(4 / 3 * len(cmlist), 2),
    )
    a = ax[0]
    a.set_yticks([0, 1])
    a.set_yticklabels(["AGG", "NGG"], fontsize=8)
    a.set_ylabel("True value", fontsize=8)
    for a, cm, title in zip(ax, cmlist, subsets):
        a.axis([-0.5, 1.5, -0.5, 1.5])
        a.set_aspect(1)
        # a.plot([0.5,0.5],[-0.5,1.5], '-k', linewidth=1)
        # a.plot([-0.5,1.5],[0.5,0.5], '-k', linewidth=1)
        a.set_xticks([0, 1])
        a.set_xticks([0.5], minor=True)
        a.set_yticks([0.5], minor=True)
        a.set_xticklabels(["NGG", "AGG"], fontsize=8)
        a.set_xlabel("Predicted Value", fontsize=8)
        a.set_title(title, fontsize=8)
        a.grid(which="minor", color="0")
        for (i, j), z in np.ndenumerate(cm):
            a.text(
                j,
                1 - i,
                "{:^5}".format(z),
                ha="center",
                va="center",
                fontsize=8,
                bbox=dict(
                    boxstyle="round", facecolor="w", edgecolor="0", linewidth=0.75
                ),
            )
    fig.tight_layout()

    if fpath is not None:
        fig.savefig(fpath, bbox_inches="tight")
        if artifact_path is not None:
            mlflow.log_artifact(str(fpath.absolute()), artifact_path)

    if return_figure:
        return fig


def train_curve(
    iters, train_acc, train_loss, val_acc, val_loss, fpath, artifact_path=None
):
    """
    Generate training/validation loss/accuracy vs training iterations curve.
    Saves to disk and logs as mlflow artifact.

    Parameters
    ----------
    iters, train_acc, train_loss, val_acc, val_loss: ndarray
        array containing number of iterations at each checkpoint,
        training accuracy, training loss, validation accuarcy, and validation loss
        at each checkpoint
    fpath: Path
        path to save and log mlflow artifact with

    artifact_path: str or None
        if None, figure is not logged as artifact
        else:
         - fpath must not be none
         - if str, path in artifact directory to log artifact to with mlflow


    Returns
    -------
    None
    """
    fpath = Path(fpath)
    c1, c2 = (0.545, 0.168, 0.886), (0.101, 0.788, 0.219)  # line rgb colors
    fig, ax = plt.subplots(1, 2, dpi=150, figsize=(6, 2.5), facecolor="w")

    a = ax[0]  # subplot for losses
    a.plot(iters, train_loss, ":*", color=c2, label="train")
    a.plot(iters, val_loss, "-.+", color=c1, label="val")
    a.set_xlabel("iterations")
    a.set_ylabel("loss")

    a = ax[1]
    a.plot(iters, train_acc, ":+", color=c2, label="train")
    a.plot(iters, val_acc, "-.*", color=c1, label="val")
    a.set_xlabel("iterations")
    a.set_ylabel("accuracy")
    a.legend()
    fig.tight_layout()
    fig.savefig(fpath, bbox_inches="tight")
    if artifact_path is not None:
        mlflow.log_artifact(str(fpath.absolute()), artifact_path)
    return


def regression_results_plot(
    gt_train, yp_train, gt_val, yp_val, gt_test, yp_test, y_min=0, y_max=1, title=""
):
    """
    Plots predicted vs ground truth values for train, validation, and testing sets on
    separate subplots with y=x overlayed in the background for reference.

    It is common to scale y data for regression y_scale = (y-y_min)/(y_max-y_min), but
    it is often desirable to view the data in its original scale during visualization.
    Thus, passing y_min and y_max, the min and max y values used to scale the data,
    will result in data being returned to its original scale:
    y_original = y_scale*(y_max-y_min) + y_min
    The default values, 0 and 1, do not affect the scale.

    Parameters
    ----------
    gt_train, yp_train, gt_val, yp_val, gt_test, yp_test: array like
        1d array corresponding to ground truth (gt) or predicted (yp) values
        for training, validation, and testing sets
    y_min, y_max: int or float
        used to transform scaled data back to its original scale, as described above.
        Default values y_min=0, y_max=1 leave the data as-is.

    title: str
        Title of entire figure (ie not individual subplots)
    Returns
    -------
    fig: figure
        plotly figure object displaying results in the format described above.
    """

    def unscale(y_scale, y_max_scale=y_max, y_min_scale=y_min):
        return (y_scale * (y_max_scale - y_min_scale)) + y_min_scale

    gt_train = unscale(gt_train)
    yp_train = unscale(yp_train)
    gt_val = unscale(gt_val)
    yp_val = unscale(yp_val)
    gt_test = unscale(gt_test)
    yp_test = unscale(yp_test)

    min_value = min(gt_train)
    max_value = max(gt_train)
    for subset in (yp_train, gt_val, yp_val, gt_test, yp_test):
        min_value = min(min_value, min(subset))
        max_value = max(max_value, max(subset))

    fig = plotly.subplots.make_subplots(
        1,
        3,
        shared_yaxes="rows",
        subplot_titles=["Train", "Validation", "Test"],
        shared_xaxes="all",
    )

    ref = go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        line={"color": "rgb(100,100,100)", "dash": "dash"},
        mode="lines",
        name="y=x",
        legendgroup=1,
    )

    for i in range(3):
        fig.add_trace(ref, row=1, col=i + 1)
        ref.update(
            showlegend=False
        )  # prevent label from showing on legend multiple times

    color = "purple"
    fig.add_trace(
        go.Scatter(
            x=gt_train,
            y=yp_train,
            mode="markers",
            legendgroup=2,
            name="results",
            showlegend=True,
            line={"color": color},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=gt_val,
            y=yp_val,
            mode="markers",
            legendgroup=2,
            name="results",
            showlegend=False,
            line={"color": color},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=gt_test,
            y=yp_test,
            mode="markers",
            legendgroup=2,
            name="results",
            showlegend=False,
            line={"color": color},
        ),
        row=1,
        col=3,
    )
    for i in range(1, 4):
        fig.layout["xaxis{0}".format(i)].title = "Ground truth"
    fig.layout["yaxis1"].title = "Predicted"
    fig.layout.title = title

    fig.update_layout(hovermode="x unified")

    return fig
