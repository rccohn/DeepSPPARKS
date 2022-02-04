"""
Feature ablation
remove individual features to measure impact on performance.
When there are small numbers of fetarures, this can be the easiest way to
directly measure which features are most important to the model, providing
some level of interpretability.
"""

from data import Dataset
import mlflow
from sgc_model import SGCNet, train_loop
from pathlib import Path
from deepspparks.utils import load_params
import itertools
from torch.optim import Adam
from torch_geometric.data import Data
from itertools import combinations
import torch
from copy import deepcopy


def format_targets(data, aggregator, thresh):
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

    thresh: int or float
        y values greater than threshold are assigned target values of 1 for
        classification (ie abnormal grain growth) Otherwise, they are assigned
        values of 0 (ie normal growth).

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

    data_formatted.y = (y_aggr > thresh).long()
    return data_formatted


def inner_run(
    k,
    dataset,
    data_train,
    lr,
    decay,
    thresh,
    data_val,
    data_test,
    params,
    artifact,
    feature_mask,
):
    """
    Helper function to reduce clutter/indentation level of parameter sweep.
    Logs training parameters, metrics, and artifacts to mlflow.

    Parameters
    ----------
    k: int
        number of iterations of message passing for SGC
    dataset: Dataset object
        dataset._log() will be called to log relevant dataset parameters to mlflow
    data_train: torch_geometric Data object
        contains training data (x, edgelist, y, candidate_mask)

    lr, decay: float
        learning rate and adam weight decay used during training
    thresh: int or float
        threshold for abnormal grain growth
    data_val, data_test
        validation and testing data (x, edgelist, y, candidate_mask)
    params: dict
        paramaters for experiment (from load_params())
    artifact: path
        path to local directory to store artifacts
    feature_mask: torch Tensor
        boolean tensor where

    Returns
    -------
    best_results: dict
        dictionary with best results from run

    """
    # converts boolean mask to string of 1/0 values for easier viewing
    ablation_str = "".join([str(x) for x in feature_mask.long().tolist()])
    with mlflow.start_run(run_name="ablation {}".format(ablation_str), nested=True):
        dataset._log()
        print(
            "running experiment for"
            "thresh={} k={}, lr={}, decay={}, features={}".format(
                thresh, k, lr, decay, feature_mask
            )
        )
        model = SGCNet(k=k, data=data_train)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
        mlflow.log_params(
            {
                "learning_rate": lr,
                "weight_decay": decay,
                "loss": "nll",
                "optimizer": "Adam",
                "cgr_thresh": thresh,
                "feature_mask": ablation_str,
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

        best_results = {
            "best_val_loss": train_metrics["val_loss"],
            "best_params": {
                "cgr_thresh": thresh,
                "k": k,
                "learning_rate": lr,
                "weight_decay": decay,
                "optimizer": "Adam",
            },
            "best_metrics": train_metrics,
            "best_state": sd,
        }
        return best_results


def main():
    param_file = "/root/inputs/params.yaml"
    print("loading params")
    params = load_params(param_file)  # parse experiment parameters
    artifact = Path(
        "/", "root", "artifacts"
    )  # path to save artifacts before logging to mlflow artifact repository

    # when running with mlflow project, run_name is not actually used, since the
    # project generates a run
    # ID associated with the project. Instead, we set the runName tag manually
    # with mlflow.set_tag()
    with mlflow.start_run(nested=False):
        mlflow.set_tag("mlflow.runName", "SGC-feature-ablation")
        mlflow.log_artifact(param_file)

        # initialize and log dataset
        dataset = Dataset(params["mlflow"]["dataset_name"])
        dataset.process(force=params["force_process_dataset"])
        dtrain, dval, dtest = dataset.train, dataset.val, dataset.test
        n_feat_max = dtrain.x.shape[1]
        feature_mask_template = torch.zeros((n_feat_max,), dtype=torch.bool)

        print("starting train loop")
        # best_results_all = []
        for thresh in params["cgr_thresh"]:
            data_train_all = format_targets(dtrain, params["repeat_aggregator"], thresh)
            data_val_all = format_targets(dval, params["repeat_aggregator"], thresh)
            data_test_all = format_targets(dtest, params["repeat_aggregator"], thresh)

            # number of features to preserve
            for n_feat in range(params["min_n_feat"], params["max_n_feat"] + 1):
                # combinations of features
                for c in combinations(list(range(n_feat_max)), n_feat):
                    # select all possible combinations of x with n_feat features
                    # filter out features
                    feature_mask = torch.clone(feature_mask_template)
                    feature_mask[list(c)] = 1

                    data_train = deepcopy(data_train_all)
                    data_val = deepcopy(data_val_all)
                    data_test = deepcopy(data_test_all)

                    data_train.x = data_train.x[:, feature_mask]
                    data_val.x = data_val_all.x[:, feature_mask]
                    data_test.x = data_test_all.x[:, feature_mask]

                    for k, lr, decay, in itertools.product(
                        params["k"],
                        params["optimizer"]["lr"],
                        params["optimizer"]["decay"],
                    ):
                        _ = inner_run(
                            k,
                            dataset,
                            data_train,
                            lr,
                            decay,
                            thresh,
                            data_val,
                            data_test,
                            params,
                            artifact,
                            feature_mask,
                        )

                        # TODO figure out if it makes sense to aggregate
                        #  results from all runs here, or
                        #     if it is better to do this externally.


if __name__ == "__main__":
    main()
