# read desired run/model from params
# get model with mlflow.<package>.load_model
# runs:/<mlflow_run_id>/run-relative/path/to/model
import mlflow

# from pathlib import Path
import numpy as np
from pathlib import Path
from deepspparks.utils import load_params
from deepspparks.visualize import agg_cm
from data import Dataset
from sklearn.metrics import confusion_matrix
import pandas as pd
import plotly.express as px


def main():
    param_file = "/root/inputs/params.yaml"
    print("parsing params")
    params = load_params(param_file)  # parse experiment parameters
    # artifact = Path(
    #    "/", "root", "artifacts"
    # )  # path to save artifacts before logging to mlflow artifact repository

    # parse params
    # load model
    # mlflow.pytorch.load_model()
    # generate predictions for train, val, test datasets
    # generate confusion matrices, classification reports, save as txt/json,
    # not figure/log (can still generate plots with standardized functions)
    # measure accuracy vs k, save results as txt and generate plot

    with mlflow.start_run(run_name="sgc-eval", nested=False):
        mlflow.log_artifact(param_file)

        best_run = mlflow.get_run(params["runs"]["best"])
        thresh = float(best_run.data.params["cgr_thresh"])

        dataset = Dataset(params["mlflow"]["dataset_name"])
        dataset.process(force=params["force_process_dataset"])

        mlflow.log_params(
            {
                "run_id_best": params["model"]["best"],
                "run_ids_k": ",".join(params["acc_vs_k"]),
            }
        )

        model = mlflow.pytorch.load_model(params["model"]["best"])

        cmats = []
        # first, evaluate best model on data
        for d, label in zip(
            (dataset.train, dataset.val, dataset.test), ("train", "val", "test")
        ):
            # get y_gt
            y_gt = (d.y > thresh).detach().numpy()

            # get y_pred
            y_pred = model.predict(d.x)[d.candidate_mask].detach().numpy()

            # get confusion matrix
            cmats.append(confusion_matrix(y_gt, y_pred))

        # log confusion matrices
        artifact = Path("/", "root/", "artifacts/")
        savepath = artifact / "confusion_mats.npy"
        np.save(savepath, cmats, allow_pickle=False)
        mlflow.log_artifact(str(savepath), artifact_path="best_model/")

        # plot confusion matrices and save to mlflow
        # TODO modify agg_cm to save raw values for cm (at least optionally?)
        agg_cm(
            cmats,
            return_figure=False,
            fpath=artifact / "confusion_matrices.png",
            artifact_path="best_model/",
        )

    # loop through uris for k=1,2,3,4
    # TODO finish this
    k_vals = []
    accs = {"train": [], "val": [], "test": []}
    for model_uri in params["acc_vs_k"]:
        model = mlflow.pytorch.load_model(model_uri)
        k_vals.append(model.k)
        cmats = []
        for d, label in zip(
            (dataset.train, dataset.val, dataset.test), ("train", "val", "test")
        ):
            # get y_gt
            y_gt = (d.y > thresh).detach().numpy()

            # get y_pred
            y_pred = model.predict(d.x)[d.candidate_mask].detach().numpy()

            # get confusion matrix
            cmats.append(confusion_matrix(y_gt, y_pred))

            accs["label"].append((y_pred == y_gt).sum() / len(y_gt))

        # log confusion matrices
        artifact = Path("/", "root/", "artifacts/")
        savepath = artifact / "confusion_mats.npy"
        np.save(savepath, cmats, allow_pickle=False)
        mlflow.log_artifact(
            str(savepath), artifact_path="acc_vs_k/k={}".format(model.k)
        )

        agg_cm(
            cmats,
            return_figure=False,
            fpath=artifact / "confusion_matrices.png",
            artifact_path="best_model/",
        )

    # plot accs vs k
    df = pd.concat(
        [
            pd.DataFrame({"accuracy": v, "dataset": [k for _ in v]})
            for k, v in accs.items()
        ]
    )

    fig = px.bar(df, x="k", y="accuracy", color="dataset", barmode="group")
    fig.update_layout(font_size=18, hover_mode="x unified")
    figpath = artifact / "acc_vs_k.html"
    fig.write_html(figpath)
    dfpath = artifact / "acc_vs_k.csv"
    df.to_csv(dfpath)

    for path in (figpath, dfpath):
        mlflow.log_artifact(str(path), artifact_path="acc-vs-k/")


if __name__ == "__main__":
    main()
