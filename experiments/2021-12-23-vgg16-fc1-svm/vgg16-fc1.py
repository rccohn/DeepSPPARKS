from data import Dataset
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
import pandas as pd
from pathlib import Path
from src.utils import parse_params
from src.visualize import agg_cm
from src.metrics import log_classification_report

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

import joblib


def main():
    print('parsing params')
    params = parse_params('params.yaml')
    paths = params['paths']
    
    for p in (paths['artifact'], paths['processed']):
        os.makedirs(p, exist_ok=True)

    print('configuring MLflow')
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])
    
    pretrained_model_path = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    assert Path(pretrained_model_path).is_file()
    for crop in (0, 1):  # False, True
        with mlflow.start_run(nested=False):
            print('logging experiment files')
            mlflow.log_param('crop', crop)
            # log experiment files
            for f in Path(__file__).parent.glob('*'):
                if not f.name.startswith('.') and f.suffix in ('.py', '.sh', '.yaml', '.yml'):
                    mlflow.log_artifact(str(f.absolute()), 'source')

            dataset = Dataset(paths['dataset'],
                              paths['processed'],
                              crop,
                              params['mlflow']['dataset_name'])
            dataset.process(vgg16_path=pretrained_model_path,
                            artifact_path=paths['artifact'], force=True, log_featurizer=True)
            X_train_raw = dataset.train['X']
            for whiten, whiten_label in enumerate(('no_whiten', 'whiten')):
                with mlflow.start_run(nested=True):
                    pca = PCA(n_components=min(X_train_raw.shape),
                              svd_solver='full', whiten=bool(whiten))
                    pca.fit(X_train_raw)
                    fname = paths['artifact'] / 'pca-{}.joblib'.format(whiten_label)
                    joblib.dump(pca, fname)
                    mlflow.log_artifact(str(fname), 'models')
                    # variance vs components plot
                    pca_var = pca.explained_variance_ratio_.cumsum()
                    # save figure/log as artifact to /figures
                    fig, ax = plt.subplots(dpi=150)
                    ax.plot(range(1,len(pca_var)+1), pca_var, '-k')
                    ax.plot(range(1,len(pca_var)+1), pca.explained_variance_ratio_, '-.k')
                    fig.tight_layout()
                    figpath = paths['artifact'] / 'pca-{}-variance.png'.format(whiten_label)
                    fig.savefig(figpath, bbox_inches='tight')
                    mlflow.log_artifact(str(figpath), 'figures')

                    # data
                    X_train_full = pca.transform(X_train_raw)
                    X_val_full = pca.transform(dataset.val['X'])
                    X_test_full = pca.transform(dataset.test['X'])

                    y_train = dataset.train['y'].astype(int)
                    y_val = dataset.val['y'].astype(int)
                    y_test = dataset.test['y'].astype(int)

                    # fit pca once, and use subsets of data for individual experiments

                    # don't use tracking server to store results for each of these in a run
                    # instead,
                    results = []
                    best_model = None
                    best_val_acc = -1.
                    best_n_components = 0
                    for var in (25, 50, 60, 65, 70, 75, 80, 90, 100):  # PCA explained variance
                        n_components = np.argmax(pca_var * 100 >= var) + 1

                        X_train = X_train_full[:, :n_components]
                        X_val = X_val_full[:, :n_components]

                        for c in np.logspace(-2, 2, 50):
                            svm = SVC(C=c, kernel="rbf", gamma="scale")
                            svm.fit(X_train, y_train)
                            yp_train = svm.predict(X_train)
                            yp_val = svm.predict(X_val)

                            train_acc = accuracy_score(y_train, yp_train)
                            val_acc = accuracy_score(y_val, yp_val)

                            results.append((n_components, var, c, train_acc, val_acc))

                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                best_model = svm
                                best_n_components = n_components

                    fpath = paths['artifact'] / 'training_results.csv'
                    headers = ('n_components', '%variance', 'svm_c', 'training_acc', 'valid_acc')
                    pd.DataFrame(data=results, columns=headers).to_csv(fpath)
                    mlflow.log_artifact(str(fpath.absolute()))

                    model_path = paths['artifact'] / "best_model.joblib"
                    joblib.dump(best_model, model_path)
                    mlflow.log_artifact(str(model_path), 'models')

                    # evaluate best model on test set
                    # log metrics, confusion matrices, acc vs c plot (for each number of components?)
                    X_train = X_train_full[:, :best_n_components]
                    X_val = X_val_full[:, :best_n_components]
                    X_test = X_test_full[:, :best_n_components]

                    yp_train = best_model.predict(X_train)
                    yp_val = best_model.predict(X_val)
                    yp_test = best_model.predict(X_test)

                    target_names = ["ngg", "agg"]
                    log_classification_report(y_train, yp_train, target_names, 'train')
                    log_classification_report(y_val, yp_val, target_names, 'validation')
                    log_classification_report(y_test, yp_test, target_names, 'test')
                    cmlist = [confusion_matrix(yt, yp) for yt, yp in ((y_train, yp_train),
                        (y_val, yp_val), (y_test, yp_test))]
                    agg_cm(cmlist, return_figure=False, fpath=paths['artifact']/'cm.png', artifact_path="figures")


if __name__ == "__main__":
    main()
