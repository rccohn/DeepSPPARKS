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
from sklearn.model_selection import KFold

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

    with mlflow.start_run(nested=False):
        print('logging experiment files')
        # log experiment files
        for f in Path(__file__).parent.glob('*'):
            if not f.name.startswith('.') and f.suffix in ('.py', '.sh', '.yaml', '.yml'):
                mlflow.log_artifact(str(f.absolute()), 'source')

        dataset = Dataset(paths['dataset'],
                          paths['processed'],
                          params['mlflow']['dataset_name'])

        dataset.process(vgg16_path=pretrained_model_path,
                        artifact_path=paths['artifact'], force=True, log_featurizer=True)
        seed = 2980476009
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        splits = tuple(kf.split(dataset.X))


        mlflow.log_param('k-fold_seed', seed)
        pca_unwhiten = []
        pca_whiten = []
        pca_all = (pca_unwhiten, pca_whiten)

        for i, s in enumerate(splits):
            X = dataset.X[s[0]]
            pca_full_unwhiten = PCA(n_components = min(X.shape), svd_solver='full', whiten=False)
            pca_full_unwhiten.fit(X)
            pca_full_whiten = PCA(n_components=min(X.shape), svd_solver='full', whiten=True)
            pca_full_whiten.fit(X)

            pca_unwhiten.append(pca_full_unwhiten)
            pca_whiten.append(pca_full_whiten)

            fname = paths['artifact'] / 'pca-{}-no-whiten.joblib'.format(i)
            joblib.dump(pca_full_unwhiten, fname)
            mlflow.log_artifact(str(fname), 'models')

            fname = paths['artifact'] / 'pca-{}-whiten.joblib'.format(i)
            joblib.dump(pca_full_whiten, fname)
            mlflow.log_artifact(str(fname), 'models')

            # whitening does not change fraction of variance explained
            pca_var = pca_full_unwhiten.explained_variance_ratio_.cumsum()
            fig, ax = plt.subplots(dpi=150)
            ax.plot(range(1, len(pca_var) + 1), pca_var, '-k')
            ax.plot(range(1, len(pca_var) + 1), pca_full_unwhiten.explained_variance_ratio_, '-.k')
            fig.tight_layout()
            figpath = paths['artifact'] / 'pca-{}-variance.png'.format(i)
            fig.savefig(figpath, bbox_inches='tight')
            mlflow.log_artifact(str(figpath), 'figures')

        with mlflow.start_run(nested=True):
            # fit pca once, and use subsets of data for individual experiments

            # don't use tracking server to store results for each of these in a run

            # instead,
            results_header = ("n_components", "% variance_preserved", "svm-C",
                              'pca_whiten', "cv_fold", 'train_acc', 'valid_acc')
            results = []
            best_val_acc = -1.
            best_n_components = 0
            best_c = 0
            best_whiten = 0

            for var in (25, 50, 60, 65, 70, 75, 80, 90, 100):  # PCA explained variance
                n_components = np.argmax(pca_var * 100 >= var) + 1

                for c in np.logspace(-2, 2, 50):
                    train_accs = ([],[]) # unwhiten, whiten
                    val_accs = ([], [])
                    for i, split in enumerate(splits):
                        y_train, y_val = dataset.y[split[0]], dataset.y[split[1]]
                        for whiten in range(2):
                            pca = pca_all[whiten][i]
                            X_train = pca.transform(dataset.X[split[0]])[:,:n_components]
                            X_val =  pca.transform(dataset.X[split[1]])[:,:n_components]

                            svm = SVC(c=c, kernel='rbf', gamma='scale')
                            svm.fit(X_train)

                            ypt = svm.predict(X_train)
                            ypv = svm.predict(X_val)

                            acc_t = accuracy_score(y_train, ypt)
                            acc_v = accuracy_score(y_val, ypv)

                            train_accs[whiten].append(acc_t)
                            val_accs[whiten].append(acc_v)
                            results.append((n_components, var, c, whiten, i, acc_t, acc_v))

                    cv_accs = [np.mean(x) for x in val_accs]
                    for whiten, acc in enumerate(cv_accs):
                        if acc > best_val_acc:
                            best_val_acc = acc
                            best_c = c
                            best_whiten = whiten
                            best_n_components = 0



                df = pd.DataFrame(data=results, columns=results_header)
                results_path = Path(paths['artifact'], 'results.csv')
                df.to_csv(results_path, index_label="index")
                mlflow.log_artifact(str(results_path))

                best_results = df[np.logical_and(df['c'] == best_c, df['whiten'] == best_whiten)]
                best_results = best_results[best_results['n_components'] == best_n_components][['train_acc'],
                                                                                               ['valid_acc']]
                mlflow.log_params({'pca_n_components': best_n_components, 'pca_whiten': best_whiten,
                                   'svm_c': best_c})
                mlflow.log_metrics({'cv_train_acc': best_results['train_acc'].mean(),
                                    'cv_valid_acc': best_results['validation_acc'].mean()})




if __name__ == "__main__":
    main()
