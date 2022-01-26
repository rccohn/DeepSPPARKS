from data import Dataset
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from deepsppark.utils import parse_params
from deepsppark.visualize import agg_cm, scree_plot

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold

import joblib


def main():
    print('Setting up experiment')
    params = parse_params('/root/inputs/params.yaml')
    artifact = Path('/root', 'artifacts')

    with mlflow.start_run(run_name='NEU CV baseline', nested=False):
        mlflow.set_tag('mlflow.runName', 'NEU CV baseline')  # mlflow project run ignores name, so we set it manually
        print(params['mlflow']['dataset_name'])
        dataset = Dataset(params['mlflow']['dataset_name'])
        dataset.process()

        kfp = params['kfold']
        mlflow.log_params({'kfold_{}'.format(k): v for k, v in kfp.items()})
        # seed = 2980476009 # old defaults
        # n_splits=5 # old defaults
        kf = KFold(n_splits=kfp['n_splits'], random_state=kfp['seed'], shuffle=True)
        splits = tuple(kf.split(dataset.X))

        pca_unwhiten = []
        pca_whiten = []
        pca_all = (pca_unwhiten, pca_whiten)

        for i, s in enumerate(splits):
            X = dataset.X[s[0]]
            pca_full_unwhiten = PCA(n_components=min(X.shape), svd_solver='full', whiten=False)
            pca_full_unwhiten.fit(X)
            pca_full_whiten = PCA(n_components=min(X.shape), svd_solver='full', whiten=True)
            pca_full_whiten.fit(X)

            pca_unwhiten.append(pca_full_unwhiten)
            pca_whiten.append(pca_full_whiten)

            fname = artifact / 'pca-{}-no-whiten.joblib'.format(i)
            assert fname.parent.is_dir(), print(artifact)
            joblib.dump(pca_full_unwhiten, fname)
            mlflow.log_artifact(str(fname), 'models/pca')

            fname = artifact / 'pca-{}-whiten.joblib'.format(i)
            joblib.dump(pca_full_whiten, fname)
            mlflow.log_artifact(str(fname), 'models/pca')

            # whitening does not change fraction of variance explained
            figpath = artifact / 'pca-scree-{}.html'.format(i)
            fig = scree_plot(pca_full_unwhiten.explained_variance_ratio_)
            fig.write_html(figpath)
            mlflow.log_artifact(str(figpath), 'figures/pca')

        # assume pca variance vs number of components is similar for all cv splits
        # this can be verified by looking at scree plots
        pca_var = pca_full_unwhiten.explained_variance_ratio_.cumsum()
        # fit pca once, and use subsets of data for individual experiments

        # metrics to store as artifact (to avoid making too many runs)
        results_header = ("n_components", "% variance_preserved", "svm-C",
                          'pca_whiten', "cv_fold", 'train_acc', 'valid_acc')
        results = []
        best_val_acc = -1.
        best_n_components = 0
        best_c = 0
        best_whiten = 0
        best_models = []
        # (25, 50, 60, 65, 70, 75, 80, 90, 100)
        for var in params['pca_var']:  # PCA explained variance
            # assume n_components same for all folds, this can be checked with scree plots
            n_components = np.argmax(pca_var * 100 >= var) + 1

            for c in np.logspace(-2, 2, params['svm_num_c']):
                for whiten in range(2):
                    train_accs = []
                    val_accs = []
                    models = []
                    for i, split in enumerate(splits):
                        y_train, y_val = dataset.y[split[0]], dataset.y[split[1]]

                        pca = pca_all[whiten][i]
                        X_train = pca.transform(dataset.X[split[0]])[:, :n_components]
                        X_val = pca.transform(dataset.X[split[1]])[:, :n_components]

                        svm = SVC(C=c, kernel='rbf', gamma='scale')
                        svm.fit(X_train, y_train)
                        models.append(svm)

                        ypt = svm.predict(X_train)
                        ypv = svm.predict(X_val)

                        acc_t = accuracy_score(y_train, ypt)
                        acc_v = accuracy_score(y_val, ypv)

                        train_accs.append(acc_t)
                        val_accs.append(acc_v)
                        results.append((n_components, var, c, whiten, i, acc_t, acc_v))

                    acc = np.mean(val_accs)
                    if acc > best_val_acc:
                        best_val_acc = acc
                        best_c = c
                        best_whiten = whiten
                        best_n_components = n_components
                        best_models = models

        # results_header = ("n_components", "% variance_preserved", "svm-C",
        #               'pca_whiten', "cv_fold", 'train_acc', 'valid_acc')

        df = pd.DataFrame(data=results, columns=results_header)
        results_path = artifact / 'results.csv'
        df.to_csv(results_path, index_label="index")
        mlflow.log_artifact(str(results_path), artifact_path='results')
        results_path = artifact / 'results.html'
        df.to_html(results_path)
        mlflow.log_artifact(str(results_path), artifact_path='results')
        best_results = (df[(df['n_components'] == best_n_components) & (df['pca_whiten'] == best_whiten)
                           & (np.isclose(df['svm-C'], best_c))]).sort_values('cv_fold')

        # contains all folds from single x-val trial
        assert best_results['cv_fold'].tolist() == list(range(kfp['n_splits']))


        mlflow.log_params({'pca_n_components': best_n_components, 'pca_whiten': best_whiten,
                           'svm_c': best_c})
        mlflow.log_metrics({'train_acc': best_results['train_acc'].mean(),
                            'valid_acc': best_results['valid_acc'].mean()})
        for _, row in best_results.iterrows():
            mlflow.log_metrics({'cv_train_acc': row['train_acc'], 'cv_val_acc':
                                row['valid_acc']}, step=int(row['cv_fold']))

        for i, (model, split) in enumerate(zip(best_models, splits)):
            mlflow.sklearn.log_model(model, 'models/SVM/cval_{}'.format(i))
            y_train, y_val = dataset.y[split[0]], dataset.y[split[1]]
            pca = pca_all[best_whiten][i]
            X_train = pca.transform(dataset.X[split[0]])[:, :best_n_components]
            X_val = pca.transform(dataset.X[split[1]])[:, :best_n_components]
            ypt = model.predict(X_train)
            ypv = model.predict(X_val)
            cm_train = confusion_matrix(y_train, ypt)
            cm_val = confusion_matrix(y_val, ypv)
            agg_cm((cm_train, cm_val), return_figure=False,
                   fpath=artifact / 'cm_cv_{}.png'.format(i),
                   artifact_path='figures/confusion_matrix/')
    # TODO log mlflow models, cv train/val acc logged with steps
    #     log final train/val acc
    #     train curves for best params
    #      plotly demo test project- copy from their examples

if __name__ == "__main__":
    main()
