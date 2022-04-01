# read desired run/model from params
# get model with mlflow.<package>.load_model
# runs:/<mlflow_run_id>/run-relative/path/to/model
import mlflow

# from pathlib import Path
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

    with mlflow.start_run(nested=False):
        mlflow.log_artifact(param_file)

        best_run = mlflow.get_run(params["runs"]["best"])
        thresh = float(best_run.data.params["cgr_thresh"])

        dataset = Dataset(params["mlflow"]["dataset_name"])
        dataset.process(force=params["force_process_dataset"])

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

        # plot confusion matrices
        agg_cm(
            cmats,
            return_figure=False,
            fpath="/root/artifacts/confusion_matrices.png",
            artifact_path="best_model/",
        )


if __name__ == "__main__":
    main()
