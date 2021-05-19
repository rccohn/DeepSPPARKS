from multiprocessing import cpu_count, get_context
from pathlib import Path
import os
from pathlib import Path
from src.graphs import Graph
from src.graph_features import default_targets
import h5py
from src.image import img_from_output
import skimage
from skimage.io import imsave
import numpy as np


def extract_graphs_spparks(src, target, n_jobs=-1, out_type='graph'):
    r"""

    Parameters
    ----------
    src: str or Path object
        path to directory of spparks results. Each folder in src
        should contain a folder called 'spparks_results', which contains
        the results for each run in the experiment.

    target: str or Path object
        path to directory to save json-extracted graphs. A corresponding
        folder to each folder in src will be created, and then the graphs
        for each run will be saved as a separate json file in the directory.

    n_jobs: int
        number of processes to use. If negative, absolute value equals number of cpu's to not use -1.
        ie n_jobs=2 will use 2 cpu's, n_jobs=-1 will use all cpu's, n_jobs=-2 will use all but 1 cpu's.

    out_type: str
        'graph': JSON-representation of Graph object will be extracted
        'img': image with candidate grain rolled to center will be extracted

    """
    out_type = out_type.lower()
    assert out_type in ('graph', 'img'), "out_type must be 'graph' or 'img'!"
    if out_type == 'graph':
        f = _ge
    elif out_type == 'img':
        f = _ie

    src = Path(src)
    target = Path(target)

    num_cpus = cpu_count()
    if n_jobs > 0:
        num_workers = min(n_jobs, num_cpus)
    elif n_jobs < 0:
        num_workers = max(num_cpus + 1 + n_jobs, 1)
    else:
        raise ("invalid n_jobs: {}".format(n_jobs))

    assert src.is_dir() and target.is_dir()

    for root in (x for x in src.glob('*') if x.is_dir()):
        subdir = Path(root, 'spparks_results')
        if subdir.is_dir():
            # create dir with same name as root in target
            target_root = Path(target, root.name)
            os.makedirs(target_root, exist_ok=True)

            runs = list(subdir.glob('*'))  # get all runs

            # pack runs with target_directory into list of tuples
            # that can be processed with map()
            pairs = [(run, target_root) for run in runs]
            #list(map(f, pairs))
            with get_context('spawn').Pool(processes=num_workers) as p:
                p.map(f, pairs)


def _ge(pair):
    """
    Graph Extractor.

    Given a tuple of (source, target) paths, where source is the
    directory of SPPARKS outputs, graph is extracted and saved as
    JSON in target.
    Parameters
    ----------
    pair: tuple
        Contains 2 elements(src, target) that are either strings or Path objects.
        src: path to SPPARKs outputs
        target: path to save graph JSON
    """
    src = pair[0]
    target = pair[1]
    try:
        g = Graph.from_spparks_out(src)
        g.to_json(Path(target, '{}.json'.format(src.name)))
    except:
        pass

def _ie(pair):
    """
    Image extractor.

    Given a tuple of (source, target) paths, where source is the
    directory of SPPARKS outputs, image of initial microstructure
    with candidate grain rolled to center is extracted and saved
    as a PNG in target.

    Parameters
    ----------
    pair: tuple
        contains 2 elements(src, target) that are either strings or Path objects.
        src: path to SPPARKS output
        target: path to save image
    """
    src = pair[0]
    target = pair[1]
    try:
        img = img_from_output(src)
        img = skimage.img_as_ubyte(img)
        root = Path(src)
        initfile = root / 'initial.dream3d'
        statsfile = root / 'stats.h5'

        assert initfile.is_file(), f'{str(initfile.absolute())} not found!'
        assert statsfile.is_file(), f'{str(statsfile.absolute())} not found!'



        init = h5py.File(initfile, 'r')
        stats = h5py.File(statsfile, 'r')
        sv = init['DataContainers']['SyntheticVolume']

        grain_labels = np.asarray(sv['CellFeatureData']['AvgQuats'])[1:]
        grain_labels = (grain_labels > 0).sum(1) - 1

        grain_sizes = np.asarray(stats['grainsize'])[:, 1:]

        gf = default_targets(grain_labels, grain_sizes)
        fname = "{}_cgr_{:.3f}_crgr_{:.3f}.png".format(src.name,
                                               gf['candidate_growth_ratio'],
                                               gf['candidate_rgr'])
        imsave(target / fname, img)
    except:
        pass


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
    """
    return np.asarray(g.edgelist).T


if __name__ == "__main__":
    root = Path('/home/ryan/Documents')

    extract_graphs_spparks(root/'test_src', root/'test_target', out_type='img')
