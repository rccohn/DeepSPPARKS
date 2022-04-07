from data import Dataset
import mlflow
from sgc_model import SGCNet, train_loop
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from deepspparks.utils import load_params, aggregate_targets
from deepspparks.visualize import agg_cm
import itertools
from sklearn.metrics import confusion_matrix
from torch.optim import Adam


def main():
    param_file = "/root/inputs/params.yaml"
    print("parsing params")
    params = load_params(param_file)  # parse experiment parameters
    if params.get("entry", "") == "eval":  # run evaluation instead of training
        import evaluation

        evaluation.main()
        raise SystemExit

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
        for thresh in params["cgr_thresh"]:
            data_train = aggregate_targets(dtrain, params["repeat_aggregator"], thresh)
            data_val = aggregate_targets(dval, params["repeat_aggregator"], thresh)
            data_test = aggregate_targets(dtest, params["repeat_aggregator"], thresh)

            for k, lr, decay, in itertools.product(
                params["k"], params["optimizer"]["lr"], params["optimizer"]["decay"]
            ):
                with mlflow.start_run(
                    run_name="training_k={}".format(k), nested=True
                ) as inner_run:
                    dataset._log()
                    print(
                        "running experiment for thresh={} k={}, lr={}, decay={}".format(
                            thresh, k, lr, decay
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
                            "cgr_thresh": thresh,
                            "k": k,
                            "learning_rate": lr,
                            "weight_decay": decay,
                            "optimizer": "Adam",
                        }
                        best_metrics_all = train_metrics
                        best_state_all = sd
                        best_run_id = inner_run.info.run_id

        # k already logged with above log_params, so we can't call model._log() to
        # log the rest)
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

        # confusion matrices
        thresh = best_params_all["cgr_thresh"]
        data_train = aggregate_targets(dtrain, params["repeat_aggregator"], thresh)
        data_val = aggregate_targets(dval, params["repeat_aggregator"], thresh)
        data_test = aggregate_targets(dtest, params["repeat_aggregator"], thresh)

        yp_train = model.predict(data_train, mask=data_train.candidate_mask)
        yp_val = model.predict(data_val, mask=data_val.candidate_mask)
        yp_test = model.predict(data_test, mask=data_test.candidate_mask)

        y_train = data_train.y[data_train.candidate_mask]
        y_val = data_val.y[data_val.candidate_mask]
        y_test = data_test.y[data_test.candidate_mask]

        cmlist = [
            confusion_matrix(gt, pred, labels=[0, 1])
            for gt, pred in zip((y_train, y_val, y_test), (yp_train, yp_val, yp_test))
        ]
        cm_path = artifact / "confusion_matrix.png"
        agg_cm(cmlist, False, cm_path, "Figures")

        # get training curves for best run
        client = mlflow.tracking.MlflowClient()
        keys = ("fit_train_acc", "fit_train_loss", "fit_val_acc", "fit_val_loss")
        run_info = {key: client.get_metric_history(best_run_id, key) for key in keys}

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.05)
        metric = run_info["fit_train_loss"]
        t1 = go.Scatter(
            x=[y.step for y in metric],
            y=[y.value for y in metric],
            name="Train",
            line_color="Crimson",
            legendgroup="group1",
            showlegend=False,
        )
        metric = run_info["fit_val_loss"]
        t2 = go.Scatter(
            x=[y.step for y in metric],
            y=[y.value for y in metric],
            name="Valid",
            line_color="BlueViolet",
            legendgroup="group2",
            showlegend=False,
        )
        metric = run_info["fit_train_acc"]
        t3 = go.Scatter(
            x=[y.step for y in metric],
            y=[y.value for y in metric],
            name="Train",
            line_color="Crimson ",
            legendgroup="group1",
            showlegend=True,
        )
        metric = run_info["fit_val_acc"]
        t4 = go.Scatter(
            x=[y.step for y in metric],
            y=[y.value for y in metric],
            name="Valid",
            line_color="BlueViolet",
            legendgroup="group2",
            showlegend=True,
        )
        fig.add_trace(t1, row=1, col=1)
        fig.add_trace(t2, row=1, col=1)
        fig.add_trace(t3, row=1, col=2)
        fig.add_trace(t4, row=1, col=2)
        fig.update_layout(
            go.Layout(
                xaxis1={"title": "iteration"},
                yaxis1={
                    "title": "nll_loss",
                },
                hovermode="x unified",
            ),
            xaxis2={"title": "iteration"},
            yaxis2={"title": "accuracy"},
        )
        figpath = artifact / "train_curve.html"
        fig.write_html(artifact / "train_curve.html")
        mlflow.log_artifact(str(figpath), "Figures")


if __name__ == "__main__":
    main()
