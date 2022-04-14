import mlflow

# from pathlib import Path
import numpy as np
from pathlib import Path
from deepspparks.utils import load_params
from deepspparks.visualize import agg_cm
from data import Dataset
from sklearn.metrics import confusion_matrix


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

        thresh = float(params["cgr_thresh"][0])

        run_id = params["eval_run_id"]
        run = mlflow.get_run(run_id)
        crop = int(run.data.params["crop"])
        model = mlflow.sklearn.load_model("runs:/{}/models/SGC".format(run_id))
        mlflow.log_params({"eval_model_run_id": run_id, "crop": crop})

        dataset = Dataset(params["mlflow"]["dataset_name"], crop)
        dataset.process(force=params["force_process_dataset"])

        mlflow.log_params(
            {
                "run_id_best": params["best_run_id"],
                "run_ids_k": ",".join(params["acc_vs_k_run_ids"]),
                "repeat_aggregator": params["repeat_aggregator"],
            }
        )

        cmats = []
        # first, evaluate best model on data
        for d, label in zip(
            (dataset.train, dataset.val, dataset.test), ("train", "val", "test")
        ):

            # get y_gt
            y_gt = d.y[d.candidate_mask].astype(np.uint8) > thresh

            # get y_pred
            y_pred = model.predict(d.x)

            # get confusion matrix
            cmats.append(confusion_matrix(y_gt, y_pred))

            # log accuracy
            acc = (y_gt == y_pred).sum() / len(y_gt)
            mlflow.log_metric("{}-accuracy".format(label), acc)

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
