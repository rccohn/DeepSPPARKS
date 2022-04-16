import mlflow

# from pathlib import Path
import numpy as np
from pathlib import Path
from deepspparks.utils import load_params
from deepspparks.visualize import agg_cm
from data import Dataset
from sklearn.metrics import confusion_matrix
import joblib


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

    with mlflow.start_run(run_name="svm-eval", nested=False):
        mlflow.log_artifact(param_file)

        thresh = float(params["cgr_thresh"][0])
        run_id = params["eval_run_id"]
        run = mlflow.get_run(run_id)
        crop = int(run.data.params["crop"])
        whiten = int(run.data.params["pca_whiten"])
        n_components = int(run.data.params["pca_n_components"])
        parent_run_id = run.data.tags["mlflow.parentRunId"]
        client = mlflow.tracking.MlflowClient()

        pca_filename = "models/pca-{}whiten.joblib".format("no_" * (1 - whiten))
        client.download_artifacts(parent_run_id, pca_filename, "/root/artifacts/")
        pca = joblib.load("/root/artifacts/{}".format(pca_filename))

        model = mlflow.sklearn.load_model("runs:/{}/models/svm".format(run_id))

        dataset = Dataset(params["mlflow"]["dataset_name"], crop)
        dataset.process(force=params["force_process_dataset"])

        mlflow.log_params(
            {
                "eval_model_run_id": run_id,
                "crop": crop,
                "pca_whiten": whiten,
                "pca_n_components": n_components,
            }
        )

        cmats = []
        # first, evaluate best model on data
        for d, label in zip(
            (dataset.train, dataset.val, dataset.test), ("train", "val", "test")
        ):

            # get y_gt
            y_gt = d["y"].astype(np.uint8) > thresh

            # get y_pred
            y_pred = model.predict(pca.transform(d["X"])[:, :n_components])

            # get confusion matrix
            cmats.append(confusion_matrix(y_gt, y_pred))

            # log accuracy
            acc = (y_gt == y_pred).sum() / len(y_gt)
            mlflow.log_metric("{}-accuracy".format(label), acc)

        # log confusion matrices
        artifact = Path("/", "root/", "artifacts/")
        savepath = artifact / "confusion_mats.npy"
        np.save(savepath, cmats, allow_pickle=False)
        mlflow.log_artifact(str(savepath), artifact_path="results/")

        # plot confusion matrices and save to mlflow
        # TODO modify agg_cm to save raw values for cm (at least optionally?)
        agg_cm(
            cmats,
            return_figure=False,
            fpath=artifact / "confusion_matrices.png",
            artifact_path="results/",
        )
