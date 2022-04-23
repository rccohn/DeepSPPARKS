from data import Dataset
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from deepspparks.utils import load_params
from deepspparks.visualize import agg_cm, scree_plot
from deepspparks.metrics import log_classification_report
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

from run_paths import pca_path


def main():
    print("Setting up experiment")
    param_file = "/root/inputs/params.yaml"
    params = load_params(param_file)  # input deck

    if params.get("entry", "") == "eval":  # run evaluation instead of training
        import evaluation

        evaluation.main()
        raise SystemExit

    artifact = Path(
        "/root", "artifacts"
    )  # path to save artifacts before logging to mlflow artifact repository
    # when running with mlflow project, run_name is not actually used, since the
    # project generates a run ID associated with the project. Instead, we set the
    # runName tag manually with mlflow.set_tag()
    with mlflow.start_run(nested=False):

        mlflow.log_artifact(param_file)

        # replace with better scores as models are trained
        best_val_acc_all = -1

        # store results from all child run in parent run for convenience
        results_all = []

        for crop in params["crop_images"]:

            # both crop and uncropped datasets are used
            # so we set log=False and remember to log parameters in child runs
            # and log best parameters in parent run
            dataset = Dataset(params["mlflow"]["dataset_name"], crop, log=False)
            dataset.process(
                vgg16_path=Path("/", "root", "inputs", "pretrained_model.h5"),
                artifact_path=artifact,
                force=params["force_process_dataset"],
            )
            X_train_raw = dataset.train["X"]
            for whiten in params["pca_whiten"]:

                # fit pca once, and use subsets of data for individual experiments

                # don't use tracking server to store results for each of these in a run
                # instead,

                print(
                    "fitting pca (crop = {}, whitening = {})".format(
                        bool(crop), bool(whiten)
                    )
                )
                pca = PCA(
                    n_components=min(X_train_raw.shape),
                    svd_solver="full",
                    whiten=bool(whiten),
                )

                pca.fit(X_train_raw)

                pca_path_local = pca_path(crop, whiten, local=True)
                pca_path_artifact = pca_path(crop, whiten, local=False)

                mlflow.sklearn.save_model(pca, path=pca_path_local)
                mlflow.sklearn.log_model(pca, artifact_path=pca_path_artifact)

                # variance vs components plot
                fig = scree_plot(pca.explained_variance_ratio_)
                figpath = artifact / "pca-{}-{}-variance.html".format(
                    ("no_" * (1 - crop)) + "crop", ("no_" * (1 - whiten)) + "whiten"
                )
                fig.write_html(figpath)
                mlflow.log_artifact(str(figpath), "figures")

                pca_var = pca.explained_variance_ratio_.cumsum()

                # data
                X_train_full = pca.transform(X_train_raw)
                X_val_full = pca.transform(dataset.val["X"])
                X_test_full = pca.transform(dataset.test["X"])

                for cgr_thresh in params["cgr_thresh"]:
                    with mlflow.start_run(run_name="vgg16-svm-trial", nested=True):
                        mlflow.set_tags(
                            {
                                "dataset": dataset.dataset_name,
                                "features": dataset.feature_name,
                                "targets": dataset.target_name,
                                "model": "sklearn-svm",
                            }
                        )
                        mlflow.log_params(
                            {
                                "crop": crop,
                                "pca_whiten": whiten,
                                "cgr_thresh": cgr_thresh,
                            }
                        )

                        results = []
                        best_val_acc = -1.0

                        y_train = (dataset.train["y"] > cgr_thresh).astype(int)
                        y_val = (dataset.val["y"] > cgr_thresh).astype(int)
                        y_test = (dataset.test["y"] > cgr_thresh).astype(int)

                        # (25, 50, 60, 65, 70, 75, 80, 90, 100)
                        for var in params["pca_var"]:  # PCA explained variance
                            n_components = np.argmax(pca_var * 100 >= var) + 1

                            X_train = X_train_full[:, :n_components]
                            X_val = X_val_full[:, :n_components]

                            for c in np.logspace(
                                params["svm_min_c"],
                                params["svm_max_c"],
                                params["svm_num_c"],
                            ):
                                print(
                                    "fitting svm for crop = {}, cgr_thresh = {}, "
                                    "var={}, "
                                    "whiten={}, c={}".format(
                                        crop, cgr_thresh, var, whiten, c
                                    )
                                )
                                svm = SVC(C=c, kernel="rbf", gamma="scale")
                                svm.fit(X_train, y_train)
                                yp_train = svm.predict(X_train)
                                yp_val = svm.predict(X_val)

                                train_acc = accuracy_score(y_train, yp_train)
                                val_acc = accuracy_score(y_val, yp_val)

                                results.append(
                                    (
                                        n_components,
                                        var,
                                        c,
                                        train_acc,
                                        val_acc,
                                    )
                                )

                                if val_acc > best_val_acc:
                                    best_val_acc = val_acc
                                    best_train_acc = train_acc
                                    best_model = svm
                                    best_n_components = n_components
                                    best_c = c
                                    best_cgr_thresh = cgr_thresh
                                    best_whiten = whiten

                        # log best model and params
                        mlflow.sklearn.log_model(best_model, artifact_path="models/svm")
                        mlflow.log_params(
                            {
                                "svm_c": best_c,
                                "cgr_thresh": best_cgr_thresh,
                                "pca_n_components": best_n_components,
                            }
                        )

                        results_header = (
                            "n_components",
                            "var",
                            "svm_c",
                            "train_acc",
                            "val_acc",
                        )

                        # log results for run
                        df = pd.DataFrame(data=results, columns=results_header)
                        results_path = artifact / "results.csv"
                        df.to_csv(results_path, index_label="index")
                        mlflow.log_artifact(str(results_path), artifact_path="results")
                        results_path = artifact / "results.html"
                        df.to_html(results_path)
                        mlflow.log_artifact(str(results_path), artifact_path="results")

                        # keep track of best performing model for parent run
                        results_all.append((results, crop, whiten, cgr_thresh))
                        if best_val_acc > best_val_acc_all:  # save for outer run
                            best_val_acc_all = best_val_acc
                            best_train_acc_all = best_train_acc
                            best_model_all = best_model
                            best_n_components_all = best_n_components
                            best_crop_all = crop
                            best_c_all = best_c
                            best_thresh_all = best_cgr_thresh
                            best_whiten_all = best_whiten

                        # train/val accs vs c for different settings in child run?

                        # evaluate best model on test set log metrics, confusion
                        # matrices, acc vs c plot (for each number of components?)
                        X_train = X_train_full[:, :best_n_components]
                        X_val = X_val_full[:, :best_n_components]
                        X_test = X_test_full[:, :best_n_components]

                        yp_train = best_model.predict(X_train)
                        yp_val = best_model.predict(X_val)
                        yp_test = best_model.predict(X_test)
                        best_test_acc = accuracy_score(y_test, yp_test)

                        target_names = ["ngg", "agg"]
                        log_classification_report(
                            y_train, yp_train, target_names, "train"
                        )
                        log_classification_report(
                            y_val, yp_val, target_names, "validation"
                        )
                        log_classification_report(y_test, yp_test, target_names, "test")
                        cmlist = [
                            confusion_matrix(yt, yp)
                            for yt, yp in (
                                (y_train, yp_train),
                                (y_val, yp_val),
                                (y_test, yp_test),
                            )
                        ]
                        agg_cm(
                            cmlist,
                            return_figure=False,
                            fpath=artifact / "cm.png",
                            artifact_path="figures",
                        )

                        mlflow.log_metrics(
                            {
                                "train_acc": best_train_acc,
                                "val_acc": best_val_acc,
                                "test_acc": best_test_acc,
                            }
                        )

        df_list = []
        print("logging final results for best model/parameters")
        for results, crop, whiten, cgr_thresh in results_all:
            df = pd.DataFrame(results, columns=results_header)
            # broadcast to all rows
            df["crop"] = crop
            df["whiten"] = whiten
            df["cgr_thresh"] = cgr_thresh
            df_list.append(df)
        df = pd.concat(df_list, ignore_index=True)

        results_path = artifact / "results.csv"
        df.to_csv(results_path, index_label="index")
        mlflow.log_artifact(str(results_path), artifact_path="results")
        results_path = artifact / "results.html"
        df.to_html(results_path)
        mlflow.log_artifact(str(results_path), artifact_path="results")

        # save model as mlflow model
        mlflow.sklearn.log_model(best_model_all, artifact_path="models/svm")

        best_pca_path = pca_path(best_crop_all, best_whiten_all, True)
        pca = mlflow.sklearn.load_model(best_pca_path)
        assert pca.whiten == bool(best_whiten_all)
        dataset = Dataset(params["mlflow"]["dataset_name"], best_crop_all, log=False)
        dataset.load()
        X_train = pca.transform(dataset.train["X"])[:, :best_n_components_all]
        X_val = pca.transform(dataset.val["X"])[:, :best_n_components_all]
        X_test = pca.transform(dataset.test["X"])[:, :best_n_components_all]

        y_train = dataset.train["y"] > best_thresh_all
        y_val = dataset.val["y"] > best_thresh_all
        y_test = dataset.test["y"] > best_thresh_all

        # evaluate best model on test set log metrics, confusion matrices, acc vs c
        # plot (for each number of components?)

        yp_train = best_model_all.predict(X_train)
        yp_val = best_model_all.predict(X_val)
        yp_test = best_model_all.predict(X_test)

        best_test_acc_all = accuracy_score(y_test, yp_test)

        target_names = ["ngg", "agg"]
        log_classification_report(y_train, yp_train, target_names, "train")
        log_classification_report(y_val, yp_val, target_names, "validation")
        log_classification_report(y_test, yp_test, target_names, "test")
        cmlist = [
            confusion_matrix(yt, yp)
            for yt, yp in ((y_train, yp_train), (y_val, yp_val), (y_test, yp_test))
        ]
        agg_cm(
            cmlist,
            return_figure=False,
            fpath=artifact / "cm.png",
            artifact_path="figures",
        )

        mlflow.set_tags(
            {
                "mlflow.runName": "candidate-grains-vgg16",  # set run name manually,
                # since mlflow project cannot set it
                "dataset": dataset.dataset_name,
                "features": dataset.feature_name,
                "targets": dataset.target_name,
                "model": "sklearn-svm",
            }
        )
        mlflow.log_params(
            {
                "cgr_thresh": best_thresh_all,
                "crop": best_crop_all,
                "pca_whiten": best_whiten_all,
                "pca_n_components": best_n_components_all,
                "svm_c": best_c_all,
            }
        )

        mlflow.log_metrics(
            {
                "train_acc": best_train_acc_all,
                "val_acc": best_val_acc_all,
                "test_acc": best_test_acc_all,
            }
        )

        # additional figures/metrics?
        # accuracy vs thresh for best_n_components?


if __name__ == "__main__":
    main()
