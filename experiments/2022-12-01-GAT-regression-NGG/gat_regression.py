from deepspparks.utils import load_params
import mlflow
from data import Dataset
from deepspparks.paths import PARAM_PATH
from nn_train import train_loop
from itertools import product
import gat_models


def main():
    # classification based on mean, std growth ratio above a threshold
    print("Loading params: {}".format(PARAM_PATH))
    assert PARAM_PATH.is_file(), "param file not found!"
    params = load_params(PARAM_PATH)

    with mlflow.start_run(run_name="GAT-regression-ngg", nested=False):
        # sometimes start_run() does not set run name tag correctly...?
        mlflow.set_tag("mlflow.runName", "GAT-regression-ngg")

        # log param file
        mlflow.log_artifact(PARAM_PATH)

        dataset = Dataset(params)
        dataset.process(params["dataset"]["force_process"])

        mlflow.log_params(
            {
                "repeat_aggregator": params["dataset"]["repeat_aggregator"],
                "heads": params["model"]["heads"],
                "dropout1": params["model"]["dropout1"],
                "dropout2": params["model"]["dropout2"],
                "encoder_mode": params["encoder"]["mode"],
                "node_encoder_uri": params["encoder"].get("node_feature_model_uri"),
                "edge_encoder_uri": params["encoder"].get("edge_feature_model_uri"),
            }
        )

        best_metrics_all = {"val_loss": 1e10}
        best_params_all = {"lr": None, "decay": None, "cgr_thresh": None}

        for lr, decay, node_components, edge_components in product(
            params["optimizer"]["lr"],
            params["optimizer"]["decay"],
            params["encoder"]["node_n_components"],
            params["encoder"]["edge_n_components"],
        ):
            with mlflow.start_run(run_name="GAT-training", nested=True):

                # set the method for aggregating/thresholding targets
                # for model to produce single prediction from repeated trials
                # and number of components to use
                dataset.set_aggregator_and_ncomponents(
                    params["dataset"]["repeat_aggregator"],
                    node_n_components=node_components,
                    edge_n_components=edge_components,
                    log=True,
                )

                sample_dataset_object = dataset.train[0]

                # re-initialize model every time
                model = gat_models.GatRegressionV1(
                    node_feat=len(sample_dataset_object.x[0]),
                    edge_dim=len(sample_dataset_object.edge_attr[0]),
                    heads=params["model"]["heads"],
                    dropout1=params["model"]["dropout1"],
                    dropout2=params["model"]["dropout2"],
                )

                print(
                    "training model with lr={}, decay={}".format(
                        lr, decay
                    )
                )
                _, best_metrics = train_loop(model, dataset, lr, decay, params)
                if best_metrics["val_loss"] < best_metrics_all["val_loss"]:
                    best_metrics_all = best_metrics
                    best_params_all = {"lr": lr, "decay": decay}

        # log metrics, generate confusion matrices
        mlflow.log_params(best_params_all)
        mlflow.log_metrics(best_metrics_all)


if __name__ == "__main__":
    main()