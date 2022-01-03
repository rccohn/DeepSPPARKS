from data import Dataset
import json
import matplotlib.pyplot as plt
import mlflow
from model import SGCNet
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from src.utils import parse_params
from src.visualize import agg_cm
from src.metrics import cm, log_classification_report
from tqdm import tqdm

import torch
from torch.optim import Adam
import torch.nn.functional as F


def train_step(model, batch, optimizer, mask):
    """
    Single step of training loop.

    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    batch: torch_geometric.data.Data object
        data to train on

    optimizer: torch.optim.Optimizer object
        optimizer for model

    mask: torch.Tensor
        [N-node] element boolean tensor.
        If mask[n] == True, loss from prediction on node n is used to update gradients during training.
        Otherwise, the prediction is ignored during training.

    Returns
    -------
    None

    Other
    -------
    Updates model parameters in-place.
    """

    model.train()  # set training mode to True (enable dropout, etc)

    optimizer.zero_grad()  # reset gradients

    log_softmax = model(batch)  # call forward on model
    labels = batch.y.long()  # ground truth y labels (as class idx, NOT one-hot encoded)

    # torch.nn.functional.nll_loss(input, target)
    # input: NxC double tensor. input[i,j] contains the log-probability (ie log-softmax)
    #                             prediction for class j of sample i
    # target: N-element int tensor N[i] gives the ground truth class label for sample i
    # forward() needs all nodes for message passing
    # however, when computing loss, we only consider nodes included in mask to avoid including unwanted nodes
    # (ie validation/test nodes, non-candidate grains, etc)
    nll_loss = F.nll_loss(log_softmax[mask], labels[mask])
    nll_loss.backward()  # compute gradients
    optimizer.step()  # update model params

    return


@torch.no_grad()  # speed up predictions by disabling gradients
def mean_acc_and_loss(model, data, mask):
    """
    Computes mean accuracy and nll loss of samples after applying mask
    to select nodes to include in calculations.

    Parameters
    ----------
----------
    model: torch.nn.Module object
        model to train

    data: torch_geometric.data.Data object
        data to evaluate model on

    mask: torch.Tensor
        [N-node] element boolean tensor.
        If mask[n] == True, loss from prediction on node n is included in accuracy and loss calculations.
        Otherwise, the prediction for node n is ignored.

    Returns
    -------
    acc, loss: float (may be 0-dim torch tensor)
        mean accuracy and nll-loss of predictions
    """
    model.train(False)
    # note: avoid using model.predict() to avoid calling forward twice (we also need log probabilities for loss)
    log_softmax = model(data)[mask]

    yp = log_softmax.argmax(dim=1)  # predicted log-probabilities
    yt = data.y[mask].long()  # ground truth class labels (int, not one-hot)

    nll_loss = float(F.nll_loss(log_softmax, yt))  # nll_loss
    acc = float((yp == yt).sum() / len(yt))  # mean accuracy

    return acc, nll_loss


# TODO update docstring
def train_loop(model, data_train, data_valid, optimizer, max_iterations,
               checkpoint_iterations, artifact_root):
    """
    Training a single model with a fixed set of parameters.

    Logs various parameters to mlflow tracking server


    Parameters
    ----------
    model: torch.nn.Module object
        model to train

    data_train, data_valid: torch_geometric.data.Data
        training, validation data

    optimizer: torch.optim.Optimizer object
        optimizer for model

    max_iterations: int

    checkpoint_iterations

    artifact_root: Path object
        temp path for local artifact storage BEFORE calling mlflow.log_artifact()

    Returns
    --------
    best_model: OrderedDict
        state dict (model.state_dict()) from model with lowest validation loss

    """
    print("Training model")
    # store initial params and initial accuracy/loss for nicer training graphs
    ta, tl = mean_acc_and_loss(model, data_train, data_train.candidate_mask)  # train accuracy, loss
    va, vl = mean_acc_and_loss(model, data_valid, data_valid.candidate_mask)  # val accuracy, loss

    best_params = model.state_dict()
    artifact_path = Path(artifact_root, 'model_state_dict_checkpoint_{:04d}.pt'.format(0))
    torch.save(best_params, artifact_path)
    mlflow.log_artifact(str(artifact_path.absolute()), "model_checkpoints")
    mlflow.log_metrics({'train_acc': ta, 'train_loss': tl, 'val_acc': va, 'val_loss': vl}, step=0)
    best_val_loss = vl
    best_iter = 0
    msg = "  iteration {:>4d}/{:<4d} Train acc: {:.4f}, Val acc: {:.4f}, Train loss: {:.4f}, Val loss: {:.4f}"
    # train model. Store train/val acc/loss at desired checkpoint iterations
    iters = range(checkpoint_iterations, max_iterations + 1, checkpoint_iterations)
    t = tqdm(iters, desc=msg.format(0, max_iterations, ta, va, tl, vl), leave=True)
    for train_iter in t:
        for _ in range(checkpoint_iterations):  # train for number of iterations in each checkpoint period
            train_step(model, data_train, optimizer, data_train.candidate_mask)

        # at the end of the checkpoint period, record loss and accuracy metrics
        ta, tl = mean_acc_and_loss(model, data_train, data_train.candidate_mask)  # train accuracy, loss
        va, vl = mean_acc_and_loss(model, data_valid, data_valid.candidate_mask)  # val accuracy, loss
        savepath = Path(artifact_root, 'model_state_dict_checkpoint_{:04d}.pt'.format(train_iter))
        torch.save(model.state_dict(), savepath)
        mlflow.log_artifact(str(savepath.absolute()), "model_checkpoints")
        mlflow.log_metrics({'train_acc': ta, 'train_loss': tl, 'val_acc': va, 'val_loss': vl}, step=train_iter)

        # update progress bar
        t.set_description(msg.format(train_iter, max_iterations, ta, va, tl, vl))
        t.refresh()


        # if validation loss is lower than previous best, update best val loss and model params
        if vl < best_val_loss:
            best_iter = train_iter
            best_val_loss = vl
            best_params = model.state_dict()

    print('\nlogging model')
    # logging best model
    model.load_state_dict(best_params)
    ypt = model.predict(data_train)[data_train.candidate_mask]  # y pred train
    ypv = model.predict(data_valid)[data_valid.candidate_mask]  # y pred validataion
    gtt = data_train.y[data_train.candidate_mask]  # ground truth train
    gtv = data_valid.y[data_valid.candidate_mask]  # ground truth validation

    target_names = ["ngg", "agg"]
    log_classification_report(gtt, ypt, target_names, 'train')
    log_classification_report(gtv, ypv, target_names, 'validation')

    cmlist = [cm(model, x, x.candidate_mask) for x in (data_train, data_valid)]
    cm_path = Path(artifact_root, 'training_cm.png')
    agg_cm(cmlist, False, cm_path, 'Figures')
    mlflow.log_metric('best_iter', best_iter)  # log iteration with lowest val loss

    return best_params


def main():
    print('parsing params')
    params = parse_params('params.yaml')  # parse experiment parameters
    paths = params['paths']

    print('creating experiment dirs')
    for p in (paths['processed'], paths['artifact']):
        if not p.is_dir():
            os.makedirs(p, exist_ok=True)

    print('configuring MLflow')
    # mlflow tracking URI and experiment name
    mlflow.set_tracking_uri(params['mlflow']['tracking_uri'])
    mlflow.set_experiment(params['mlflow']['experiment_name'])

    # best model weights for each training session
    k_values = []
    best_weights = []

    with mlflow.start_run(nested=False):
        # log all experiment files
        print('logging experiment files')
        for f in Path(__file__).parent.glob('*'):
            if not f.name.startswith('.') and f.suffix in ('.py', '.sh', '.yaml', '.yml'):
                mlflow.log_artifact(str(f.absolute()), 'source')
        # initialize and log dataset
        dataset = Dataset(paths['dataset'],
                          paths['processed'],
                          params['mlflow']['dataset_name'])
        dataset.process()
        dtrain, dval, dtest = dataset.train, dataset.val, dataset.test
        print('starting train loop')
        for k in params['model']['k']:
            for lr in params['optimizer']['lr']:
                for decay in params['optimizer']['decay']:
                    print("running experiment for k={}, lr={}, decay={}".format(k, lr, decay))
                    with mlflow.start_run(nested=True,):
                        
                        model = SGCNet(k=k, data=dtrain)
                        optimizer = Adam(model.parameters(), 
                        lr=lr, weight_decay=decay)
                        mlflow.set_tag('optimizer', 'Adam')
                        mlflow.log_params({
                        'learning_rate': lr,
                        'weight_decay': decay})
                        
                        sd = train_loop(model, dtrain, dval, optimizer,
                                        params['training']['max_iter'],
                                        params['training']['checkpoint_iter'],
                                        paths['artifact'])
                        # training/validation acc/loss curve, training/validation confusion matrix
                        # and classification report
                        # can be done in train loop function

            k_values.append(k)
            best_weights.append(sd)

        artifact_root = paths['artifact']

        print('processing final results')
        k_results = []
        for k, sd in zip(k_values, best_weights):
            # recover weights from best model for each k value, and
            # generate confusion matrix for train, val, and test data
            model = SGCNet(k=k, data=dtrain, log=False)
            model.load_state_dict(sd)
            keys = ('train', 'val', 'test')
            results = {key: cm(model, v, v.candidate_mask) for key, v
                       in zip(keys, (dtrain, dval, dtest))}

            # generate figures for confusion matrices, save, and log to mlflow
            agg_cm(cmlist=results, return_figure=False, fpath=Path(artifact_root,
                                                                   'cm_k={}.png'.format(k)), artifact_path='Figures')

            # convert confusion matrix arrays to lists and dump to json
            json_path = Path(artifact_root, 'cm_k={}.json'.format(k))
            with open(json_path, 'w') as f:
                json.dump({key: v.tolist() for key, v in results.items()}, f)

            df_data = {'acc': [], 'dataset': [], 'k': []}
            for key, value in results.items():
                df_data['acc'].append(value.trace()/value.sum())  # accuracy from confusion matrix
                df_data['dataset'].append(key)
                df_data['k'].append(k)

            k_results.append(pd.DataFrame(df_data))  # add

        df = pd.concat(k_results)
        # bar graphs for train/val/test acc vs k
        fig, ax = plt.subplots(figsize=(4, 3), dpi=300, facecolor='w')
        sns.barplot(x='k', y='acc', hue='dataset', ax=ax, data=df,
                    palette=np.array(sns.color_palette('bright'))[[2, 4, 6]],
                    )
        fig.tight_layout()
        figpath = Path(artifact_root, 'acc_vs_k.png')
        fig.savefig(figpath, bbox_inches='tight')
        mlflow.log_artifact(str(figpath.absolute()), artifact_path="Figures")


if __name__ == "__main__":
    main()
