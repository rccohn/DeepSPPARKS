from data import Dataset
import mlflow
from sgc_model import SGCNet, train_loop

from pathlib import Path

from deepspparks.utils import load_params
from deepspparks.visualize import regression_results_plot


import itertools

from torch.optim import Adam
from torch_geometric.data import Data


def format_targets(
    data,
    aggregator,
):
    """
    formats dataset with repeated SPPARKS trials. Converts
    array of cgr values from repeated trials into a target for
    a machine learning model.

    Parameters
    ----------
    data: Data object
        torch_geometric Data object

    aggregator: str
        describes how repeated trials are aggregated into a single target.
        Currently only supports "mean", but can easily be expanded later
        supported aggregators are:
          - "mean": averages values

    Returns
    -------
    data_formatted: Data object
        torch_geometric Data object with updated target values.

    """
    data_formatted = Data(
        x=data.x, edge_index=data.edge_index, candidate_mask=data.candidate_mask
    )
    if aggregator == "mean":
        y_aggr = data.y.mean(1)

    data_formatted.y = y_aggr

    return data_formatted


def main():
    param_file = "/root/inputs/params.yaml"
    print("parsing params")
    params = load_params(param_file)  # parse experiment parameters
    artifact = Path(
        "/", "root", "artifacts"
    )  # path to save artifacts before logging to mlflow artifact repository

    # when running with mlflow project, run_name is not actually used, since the
    # project generates a run ID associated with the project. Instead, we set the
    # runName tag manually with mlflow.set_tag()

    with mlflow.start_run(nested=False):
        mlflow.set_tag("mlflow.runName", "cgr-SGC")
        mlflow.log_param("repeat_aggregator", params["repeat_aggregator"])
        mlflow.log_artifact(param_file)
        # initialize and log dataset
        dataset = Dataset(params["mlflow"]["dataset_name"])
        dataset.process(force=params["force_process_dataset"])
        dtrain, dval, dtest = dataset.train, dataset.val, dataset.test
        print("starting train loop")
        best_val_loss_all = 1e10

        data_train = format_targets(dtrain, params["repeat_aggregator"])
        data_val = format_targets(dval, params["repeat_aggregator"])
        data_test = format_targets(dtest, params["repeat_aggregator"])

        for k, lr, decay, in itertools.product(
            params["k"], params["optimizer"]["lr"], params["optimizer"]["decay"]
        ):
            with mlflow.start_run(
                run_name="training_k={}".format(k), nested=True
            ):  # as inner_run:
                dataset._log()
                print(
                    "running experiment for k={}, lr={}, decay={}".format(k, lr, decay)
                )
                model = SGCNet(k=k, data=data_train)
                optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
                mlflow.log_params(
                    {
                        "learning_rate": lr,
                        "weight_decay": decay,
                        "loss": "mse",
                        "optimizer": "Adam",
                    }
                )

                sd, train_metrics = train_loop(
                    model,
                    data_train,
                    data_val,
                    data_test,
                    optimizer,
                    params["training"]["max_iter"],
                    params["training"]["checkpoint_iter"],
                    artifact,
                )
                model.load_state_dict(sd)
                mlflow.pytorch.log_model(model, artifact_path="models/SGC")

                if train_metrics["val_loss"] < best_val_loss_all:
                    best_val_loss_all = train_metrics["val_loss"]
                    best_params_all = {
                        "k": k,
                        "learning_rate": lr,
                        "weight_decay": decay,
                        "optimizer": "Adam",
                        "loss": "mse",
                    }
                    best_metrics_all = train_metrics
                    best_state_all = sd
                    # best_run_id = inner_run.info.run_id

        # k already logged with above log_params, so we can't call model._log()
        mlflow.set_tags(
            {
                "model_name": model.model_name,
            }
        )
        # log best results to outer run
        mlflow.log_params(best_params_all)
        mlflow.log_metrics(best_metrics_all)
        model.load_state_dict(best_state_all)
        mlflow.pytorch.log_model(model, "best_model")

        data_train = format_targets(
            dtrain,
            params["repeat_aggregator"],
        )
        data_val = format_targets(
            dval,
            params["repeat_aggregator"],
        )
        data_test = format_targets(
            dtest,
            params["repeat_aggregator"],
        )

        yp_train = (
            model.predict(data_train, mask=data_train.candidate_mask).detach().numpy()
        )
        yp_val = model.predict(data_val, mask=data_val.candidate_mask).detach().numpy()
        yp_test = (
            model.predict(data_test, mask=data_test.candidate_mask).detach().numpy()
        )

        y_train = data_train.y[data_train.candidate_mask].detach().numpy()
        y_val = data_val.y[data_val.candidate_mask].detach().numpy()
        y_test = data_test.y[data_test.candidate_mask].detach().numpy()

        fig = regression_results_plot(
            y_train, yp_train, y_val, yp_val, y_test, yp_test, "CGR values"
        )
        figpath = artifact / "regression_results.html"
        fig.write_html(str(figpath))
        mlflow.log_artifact(str(figpath), "figures")


if __name__ == "__main__":
    main()
