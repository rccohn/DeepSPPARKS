from data import Dataset
import json
import matplotlib.pyplot as plt
import mlflow
from sgc_model import SGCNet, mean_acc_and_loss, train_loop
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from deepspparks.utils import parse_params
from deepspparks.visualize import agg_cm
from deepspparks.metrics import cm
import itertools
from torch.optim import Adam
from torch_geometric.data import Data


def main():
    print('parsing params')
    params = parse_params('/root/inputs/params.yaml')  # parse experiment parameters
    artifact = Path('/', 'root', 'artifacts')  # path to save artifacts before logging to mlflow artifact repository

    # when running with mlflow project, run_name is not actually used, since the project generates a run
    # ID associated with the project. Instead, we set the runName tag manually with mlflow.set_tag()
    with mlflow.start_run(nested=False):
        mlflow.set_tag('mlflow.runName', 'cgr-SGC')

        # initialize and log dataset
        dataset = Dataset(params['mlflow']['dataset_name'])
        dataset.process()
        dtrain, dval, dtest = dataset.train, dataset.val, dataset.test
        print('starting train loop')
        best_val_loss_all = 1e10
        for thresh in params['cgr_thresh']:
            data_train = Data(x=dtrain.x, edge_index=dtrain.edge_index,
                              y=dtrain.y > thresh, candidate_mask=dtrain.candidate_mask)
            data_val = Data(x=dval.x, edge_index=dval.edge_index,
                            y=dval.y > thresh, candidate_mask=dval.candidate_mask)
            data_test = Data(x=dtest.x, edge_index=dtest.edge_index,
                             y=dtest.y > thresh, candidate_mask=dtest.candidate_mask)

            for k, lr, decay, in itertools.product(params['k'], params['optimizer']['lr'],
                                                   params['optimizer']['decay']):
                with mlflow.start_run(run_name='training_k={}'.format(k), nested=True):

                    print("running experiment for k={}, lr={}, decay={}".format(k, lr, decay))
                    model = SGCNet(k=k, data=data_train)
                    optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
                    mlflow.log_params({'learning_rate': lr, 'weight_decay': decay,
                                       'loss': 'nll', 'optimizer': 'Adam',
                                       'cgr_thresh': thresh})

                    sd, train_metrics = train_loop(model, data_train, data_val, data_test, optimizer,
                                                   params['training']['max_iter'],
                                                   params['training']['checkpoint_iter'],
                                                   artifact)
                    model.load_state_dict(sd)
                    mlflow.pytorch.log_model(model, artifact_path='models/SGC')

                    if train_metrics['val_loss'] < best_val_loss_all:
                        best_val_loss_all = train_metrics['val_loss']
                        best_params_all = {'cgr_thresh': thresh, 'k': k, 'lr': lr, 'decay': decay}
                        best_metrics_all = train_metrics
                        best_state_all = sd

        # log best results to outer run
        mlflow.log_params(best_params_all)
        mlflow.log_metrics(best_metrics_all)
        model.load_state_dict(best_state_all)
        mlflow.pytorch.log_model(model, 'best_model')



                    # training/validation acc/loss curve, training/validation confusion matrix
                    # and classification report
                    # can be done in train loop function



if __name__ == "__main__":
    main()
