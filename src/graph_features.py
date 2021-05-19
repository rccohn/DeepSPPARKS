"""
Routines for computing features on standard Graph object.
"""
import numpy as np
from scipy.spatial.distance import cdist
import pycocotools.mask as RLE

# TODO change mobility label to 'type' or something more technically correct
def compute_node_features(mask, mob):
    #  TODO degree?
    """
    Parameters
    -----------
    mask- binary mask of grain of interest
    mob- int label of mobility

    Returns
    --------
    features: dict containing mobility label and RLE encoding of mask
    """

    features = {}

    # mask is entire size of image- select only part of image where grain is
    # located for easier computation and for packing into bitmask
    rr, cc = np.where(mask)
    mask = mask[rr.min():rr.max()+1, cc.min():cc.max()+1]

    #props = ['area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length']

    #features_ = skimage.measure.regionprops_table(mask.astype(np.uint8), properties=props)

    features['mobility_label'] = int(mob)

    # create bitmask, center grain on bitmask
    rle = RLE.encode(np.asfortranarray(mask))
    features['rle'] = rle
    return features


def compute_edge_features(labels, src, target):
    """
    labels: r x c int image where pixels are grain ids
             should be centered so src is in the middle of the image
             to prevent src or targets from being wrapped around rows or cols
    src, target: int, source, target indecies of directed edge

    properties:
    length: centroid to centroid distance between source and target
    unit vector: x and y components of unit vector pointing from centroid of source to
                centroid of target
    intersect_length (not currently used): number of pixels on source within 1 pixel distance from boundary
    """

    src_mask = labels == src
    target_mask = labels == target

    where_s = np.where(src_mask)
    where_t = np.where(target_mask)

    # (r,c) coordinate pairs of centroid determined from the mean of all points
    # note it is assumed that neither grain is wrapped around rows or columns
    centroid_source = np.mean(where_s, axis=1)
    centroid_target = np.mean(where_t, axis=1)

    # centeroid to centroid distance, squeezed because there is only 1 point
    edge_length = float(cdist(centroid_source[np.newaxis, :],
                                        centroid_target[np.newaxis, :]))

    # unit vector pointing from target to source (r,c coords)
    # (for directed graphs, neighborhood of source is defined by edges from target to source)

    v = centroid_source - centroid_target
    v = v / np.sqrt(np.sum(np.power(v, 2)))

    features = {'length': edge_length,
                'unit_vector': v, }

    return features


def compute_metadata(img, grain_labels, grain_sizes, timesteps, path):
    """
    metadata
    """

    mobility_vectors = sorted(np.unique(grain_labels, axis=0), key=lambda x: x.astype(np.bool).sum())
    mobility_vectors = tuple(zip(
                                        [f'type_{i}' for i in range(len(mobility_vectors))],
                                        [x.tolist() for x in mobility_vectors],
                                        ('candidate grain (white)', 'low mobility (blue)', 'high mobility (red)')))
    mobility_vectors = {x[0]: (x[1:]) for x in mobility_vectors}  # dict makes json saving/loading easier
    r, c = img.shape
    center_id = int(img[r//2, c//2])  # cast to python int for json compatibility
    rr, cc = np.where(img == center_id)

    metadata = {'img_size': (r, c),
                'timesteps': timesteps,  # timesteps at which values are reported from spparks
                'mobility_vectors': mobility_vectors,  # original mobilities from spparks
                'grain_sizes': grain_sizes,  # grain size vs time from spparks
                'center_id': center_id,  # id of grain located at center of image
                # min/max row/col of grain passing through center. Used to fix absolute positions
                # during reconstruction of the original graph
                'center_bounds_rrcc': np.array([rr.min(), rr.max(), cc.min(), cc.max()]),
                'path': path  # path to original spparks results directory
                }

    return metadata


def default_targets(grain_labels, grain_sizes):
        """
        Default metrics for growth of candidate grain.

        growth ratio is area(final)/area(initial)
        bulk growth ratio only includes grains that
        are not the candidate grain, and whose sizes increase during the simulation
        relative growth ratio (rgr) is candidate growth ratio / bulk growth ratio

        Parameters
        ----------
        grain_ids: ndarray
            r x c image 
        """

        grow_ratio_all = grain_sizes[-1] / grain_sizes[0]

        grow_mask = grow_ratio_all > 0

        candidate_mask = grain_labels == 0
        bulk_mask = np.logical_and(grow_mask, np.logical_not(candidate_mask))

        cgr = grow_ratio_all[candidate_mask].mean()  # candidate growth ratio
        bgr = grow_ratio_all[bulk_mask].mean()  # bulk growth ratio
        crgr = cgr / bgr  # candidate relative growth ratio

        targets = {'candidate_growth_ratio': cgr,
                    'bulk_growth_ratio': bgr,
                    'candidate_rgr': crgr}

        return targets


def featurizer(g, fedgelist=None, fnode=None, fedge=None, fgraph=None, targets=None):
    """
    Takes in graph object as input, and computes node, edge, and graph features, as well
    as targets for each graph.

    Parameters
    -----------
    g: Graph object
        graph to extract features and targets from
    fedgelist, fnode, fedge, fgraph, targets: callable or None
        Function that takes in Graph object as input, and returns the formatted edgelist,
        node features, edge features, graph features, and targets (outputs,) respectively

        If None, then returns None

    Returns
    ------------
    feat: dict
        dictionary of features with following key-value pairs:
            'edgelist': 2xn or nx2 array or tensor of formatted values, or None.
                        Node ids should be ints starting from 0 (ie can be used as indices of feature matrix.)
                        Edges should also appear in order so edgelist[i] (if edgelist is nx2)
                        or edgelist[:,i] (if edgelist is 2xn) corresponds to fedge[i] in edge
                        feature matrix.
            'fnode': 2d array or tensor of node features, or None
                     Note: Features are assumed to be sorted, so fnode[i] gives features
                     for note with index i)
            'fedge': 2d array or tensor of edge features, or None
                     Note:assumes features are sorted, so fedge[i] gives features
                     for edge associated with edge i in edgelist.
            'fgraph': 1d array or tensor of graph features, or None.
            'targets': dict, array-like, numeric, or None
                       Desired output or outputs for graph
    """
    # initialize values
    keys = ('edgelist', 'fnode', 'fedge', 'fgraph', 'targets')
    feat = {x: None for x in keys}

    for k, f in zip(keys, (fedgelist, fnode, fedge, fgraph, targets)):
        if f is not None:
            feat[k] = f(g)
    return feat

if __name__ == '__main__':

    mask = np.zeros((10, 10), np.bool)
    mask[2:4, 2:4] = True
    mob = 2

    print(node_features(mask, mob))