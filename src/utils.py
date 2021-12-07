import json
import numpy as np
from pathlib import Path

from src.graphs import Graph

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
    ldata = len(data)
    nbatch = ldata // batch_size + int(ldata % batch_size > 0)
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
    Parameters
    ----------
    None

    Returns
    ---------
    edgelist: ndarray
     edgelist in pyg format
     # TODO this was broken with later changes in graph
     #    I think it should work with replacing edgelist with g.eli
     #    but this needs to be tested
    """
    return np.asarray(g.edgelist).T


def load_json_old(inpath):
    """
    Loads Graph from old json format

    Parameters
    ----------
    inpath: str or Path object
        path to old file to load

    Returns
    -------
    G: Graph
        loaded graph object
    """
    with open(inpath, 'r') as f:
        data = json.load(f)

    md_old = data['metadata']

    metadata = {'img_size': md_old['img_size'],
                'grain_types': md_old['grain_types']}
    path = md_old['path']
    if not type(path) == list:
        path = [path]
    metadata['path'] = [Path(p) for p in path]

    timesteps = md_old['timesteps']
    if not type(timesteps[0]) == list:
        timesteps = [timesteps]

    metadata['timesteps'] = [np.array(x) for x in timesteps]

    grain_sizes = md_old['grain_sizes']
    if not type(grain_sizes[0][0]) == list:
        grain_sizes = [grain_sizes]
    grain_sizes = [np.array(x) for x in grain_sizes]
    G = Graph()
    for k,v in data['nodes'].items():
        k = int(k)
        v['rle']['counts'] = bytes(v['rle']['counts'], 'utf-8')
        node_features = {'grain_type': v['grain_type'],
                         'rle': v['rle'],
                         'grain_size': [x[:, k] for x in grain_sizes]}
        G.add_node(k, **node_features)
    for k in data['edges'].keys():
        e = [int(x) for x in k.strip('()').split(', ')]
        G.add_edge(*e)


    G.metadata = {'subgraph_metadata': [metadata],
                  'subgraph_node_ranges': [len(G.nodes)]}

    return G

if __name__ == "__main__":
    pass
