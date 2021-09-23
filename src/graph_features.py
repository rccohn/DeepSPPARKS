"""
Routines for computing features on standard Graph object.
"""
import numpy as np
from scipy.spatial.distance import cdist
import pycocotools.mask as RLE


# TODO change mobility label to 'type' or something more technically correct
def compute_node_features(mask, gtype):
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

    features['grain_type'] = int(gtype)

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

    # (r,c) coordinate of top left pixel of each node
    # note it is assumed that the grain in question is not 'wrapped' around the image
    # otherwise you have to compute all possible displacements (target is above, below,
    # left, and right of source) and select one

    rc_source = np.min(where_s, axis=1)
    rc_target = np.min(where_t, axis=1)
    dr = rc_source - rc_target
    features = {'dr': dr}

    return features


def compute_metadata(img, grain_labels, grain_sizes, timesteps, path):
    """
    metadata
    """

    grain_types = list(range(3))

    # values copied from SPPARKS directly
    v1 = 0.4472135901451111
    v2 = 0.5773502588272095
    v3 = 0.8944271802902222
    grain_types = tuple(zip(
                            [f'type_{i}' for i in range(3)],
                            [[1., 0., 0.], [v3, v1, 0.], [v2, v2, v2]],
                            ('candidate grain (white)', 'low mobility (blue)', 'high mobility (red)')))
    grain_types = {x[0]: (x[1:]) for x in grain_types}  # dict makes json saving/loading easier
    r, c = img.shape
    center_id = int(img[r//2, c//2])  # cast to python int for json compatibility
    rr, cc = np.where(img == center_id)

    metadata = {'img_size': (r, c),
                'timesteps': timesteps,  # timesteps at which values are reported from spparks
                'grain_types': grain_types,  # original mobilities from spparks
                'grain_sizes': grain_sizes,  # grain size vs time from spparks
                'center_id': center_id,  # id of grain located at center of image
                # min row/col of grain passing through center. Used to fix absolute positions
                # during reconstruction of the original graph
                'center_rc': [int(rr.min()), int(cc.min())],
                'path': path  # path to original spparks results directory
                }

    return metadata


if __name__ == '__main__':
    pass
