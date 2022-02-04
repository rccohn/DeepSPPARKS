import numpy as np
import os
from pathlib import Path
import re
import yaml
import mlflow


def batcher(data, batch_size=3, min_size=2):
    r"""
    Split list into batches of approximately equal length.

    Useful for dividing sets of graphs into batches.

    Parameters
    ----------
    data: list-like
        list of items to be divided into batches
    batch_size: int
        length of each batch (sub-list) in the new list
    min_size: int
        If batches cannot be divided evenly (ie batches of 3 from list length 10), specifies the minimum
        length of each batch (ie the leftover batch.) If the last batch contains fewer than min_size elements,
        it will be appended to the previous batch. For example, batcher(list(range(5)), batch_size=2, min_size=2)
        yields [[0, 1], [2, 3, 4]], to prevent the last batch containing only 1 element (4).

    Returns
    -------
    batches: list
        list whose elements are elements of data divided into batches


    Examples
    --------
    >>> batcher(list(range(5)), 2, 2)
    [[0, 1], [2, 3, 4]]

    """

    batches = []
    n = len(data)
    nbatch = n // batch_size + int(n % batch_size > 0)
    for j in range(nbatch):
        i1 = j * batch_size
        i2 = i1 + batch_size
        batches.append(data[i1:i2])
    if len(batches) > 1 and len(batches[-1]) < min_size:
        batches[-2].extend(batches[-1])
        batches = batches[:-1]

    return batches


def pyg_edgelist(g):
    """
    Converts edgelist to pytorch-geometric compatible form.

    Parameters
    ----------
    g: Graph object
        graph to extract edgelist from

    Returns
    ---------
    edgelist: ndarray
     edgelist in pyg format
     # TODO this was broken with later changes in graph
     #    I think it should work with replacing edgelist with g.eli
     #    but this needs to be tested
    """
    return np.asarray(g.eli).T


def load_params(in_path):
    """
    Parses parameter input yaml file for experiments.

    Substitutes shell environment variables (ie $x, ${x}) where appropriate

    Format of yaml file is as follows:
    mlflow: dict
        mlflow tracking parameters (experiment name, tracking uri, etc)
    paths: dict
         maps paths (raw, pre-processed, processed) to paths
    model, optimizer: dict(list):
        list of model and optimizer parameters to use during training. All possible
        combinations of parameters will be used.
    training: dict
        other training parameters (like max training iterations in each trial
        and number of iterations per checkpoint interval)

    Parameters
    ----------
    in_path: str or Path object
        path to params yaml file

    Returns
    -------
    params: dict
        parsed dictionary of parameters

    """
    in_path = Path(in_path)
    with open(in_path, 'r') as f:
        data = yaml.safe_load(f)
    params = _parse_params(data, None)

    if 'paths' in params.keys():
        params['paths'] = _str2path(params['paths'])
        _make_paths(params['paths'])

    if 'mlflow' in params.keys():
        mlf_params = params['mlflow']
        if 'tracking_uri' in mlf_params.keys():
            mlflow.set_tracking_uri(mlf_params['tracking_uri'])
        if 'experiment_name' in mlf_params.keys():
            mlflow.set_experiment(mlf_params['experiment_name'])

    return params


def _parse_params(params, sub_dict=None):
    """
    helper function for parse_params

    Parameters
    ----------
    params: dict
        original params dictionary
    sub_dict: dict or None
        dictionary that may be a value in one of the keys of params,
        or a value of a sub_dict in params.
        If none, then params is used.

    Returns
    ----------
    params_update: dict
        parameters with os and params expressions substituted
    """
    if sub_dict is None:
        sub_dict = params

    # regex for matching environment variables
    env_var_pattern = re.compile(r'\${?[a-zA-Z_]+[a-zA-z0-9_]*}?')

    for k, v in sub_dict.items():
        if type(v) == dict:
            sub_dict[k] = _parse_params(params, v)
        elif type(v) == str:
            matches = [x for x in env_var_pattern.finditer(v)]
            if len(matches):
                for m in matches:
                    g = m.group()
                    new = g.strip('$').strip('{}') # extract varialbe name
                    v = v.replace(g, os.environ[new])
            
            sub_dict[k] = v

        else:
            sub_dict[k] = v

    return sub_dict


def _str2path(v):
    """
    Helper function to convert paths in params from string to Path object

    Parameters
    ----------
    v: None, String, or dict
        should be value of params['path'] after parsing

    Returns
    -------
    vpath: None, Path, or dict
        new

    """
    t = type(v)
    if t is None:
        return
    elif t == str:
        return Path(v)
    elif t == dict:
        return {k: _str2path(x) for k, x in v.items()}

def _make_paths(v):
    """
    Helper function to make dirs in params paths.

    Parameters
    ----------
    v: None, Path, or dict

    Returns
    -------

    """
    t = type(v)
    if t is None:
        return
    elif t == Path:
        os.makedirs(v, exist_ok=True)
    elif t == dict:
        # recursively make all paths in entries of v
        {_make_paths(vv) for vv in v.values()}
        return

if __name__ == "__main__":
    pass
