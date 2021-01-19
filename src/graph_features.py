import numpy as np
import skimage
import skimage.measure
import skimage.morphology
from scipy.spatial.distance import cdist


def compute_node_features(mask, mob, degree):
    #  TODO degree?
    """
    mask- binary mask of grain of interest
    mob- int label of mobility
    degree- int number of neighbors

    features: area, perimeter, major/minor axis, equivalent diameter
    """

    temp = np.zeros(3)
    temp[mob] = 1

    props = ['area', 'perimeter', 'equivalent_diameter', 'major_axis_length', 'minor_axis_length']

    features = skimage.measure.regionprops_table(mask.astype(np.uint8), properties=props)

    features['mobility_label'] = temp
    features['degree'] = np.array([degree])

    return features

# TODO since only difference between edges AB and BA is the components of the unit vector,
#      look into calling this only once per pair of edges
def compute_edge_features(labels, src, target):
    """
    labels: r x c int image where pixels are grain ids
    src, target: int, source, target indecies of directed edge

    properties:
    length: centroid to centroid distance between source and target
    theta: angle relative to X of vector pointing from source to target
    intersect_length: number of pixels on source within 1 pixel distance from boundary
    """
    src_mask = labels == src
    target_mask = labels == target

    where_s = np.where(src_mask)
    where_t = np.where(target_mask)

    # (r,c) coordinate pairs of centroid determined from the mean of all points
    centroid_source = np.mean(where_s, axis=1)
    centroid_target = np.mean(where_t, axis=1)

    # centeroid to centroid distance, squeezed because there is only 1 point
    edge_length = np.array([float(cdist(centroid_source[np.newaxis, :],
                        centroid_target[np.newaxis, :]))])

    # unit vector pointing from target to source (r,c coords)
    # (for directed graphs, neighborhood of source is defined by edges from target to source)

    v = centroid_source - centroid_target
    v = v / np.sqrt(np.sum(np.power(v, 2)))



    # select points on edge of each mask by removing the inside of each mask (binary erosion)
    selem = np.ones((3, 3), np.uint8)

    # inverse eroded masks (used to only select edge pixels from original masks)
    src_mask_erode = np.logical_not(skimage.morphology.binary_erosion(src_mask, selem=selem))
    target_mask_erode = np.logical_not(skimage.morphology.binary_erosion(target_mask, selem=selem))

    # coordinates of edge pixels
    where_se = np.stack(np.where(np.logical_and(src_mask, src_mask_erode)), axis=1)
    where_te = np.stack(np.where(np.logical_and(target_mask, target_mask_erode)), axis=1)

    # number of pixels on boundary between source and target grains
    # compute distance from all edge pixels in source to all edge pixels in target
    # find the minimum distance (ie distance to closest pixel)
    # count the number of edge pixels (minimum distance == 1.0 pixels)
    cd = cdist(where_se, where_te)
    n_bound_1 = (cd.min(axis=0) == 1.0).sum()
    n_bound_2 = (cd.min(axis=1) == 1.0).sum()
    n_bound = np.array([max(n_bound_1, n_bound_2)])

    features = {'length': edge_length,
                'unit_vector': v,
                'n_bound': n_bound}

    return features


def compute_graph_features(img, grain_ids, grain_sizes, timesteps):
    """
    compute useful graph features
    """


    metadata = {'img_size': img.shape, 'timesteps': timesteps, 'grain_types': {'type_0': 'candidate grain (white)',
                                                                               'type_1': 'low mobility (blue)',
                                                                               'type_2': 'high mobility (red)'
                                                                               }
                }

    grow_ratio_all = grain_sizes[-1] / grain_sizes[0]

    grow_mask = grow_ratio_all > 0

    bulk_mask = np.logical_and(grow_mask, grain_ids != 0)
    candidate_mask = grain_ids == 0

    cgr = grow_ratio_all[candidate_mask].mean()  # candidate growth ratio
    bgr = grow_ratio_all[bulk_mask].mean()  # bulk growth ratio
    crgr = cgr / bgr  # candidate relative growth ratio

    high_mobility_fraction = (grain_ids == 2).sum() / len(grain_ids)
    high_mobility_area_fraction = grain_sizes[0, grain_ids == 2].sum()/grain_sizes[0, :].sum()
    features = {'candidate_growth_ratio': cgr,
                'bulk_growth_ratio': bgr,
                'candidate_rgr': crgr,
                'metadata': metadata,
                'fraction_red_grains': high_mobility_fraction,
                'area_fraction_red_grains': high_mobility_area_fraction}

    return features














if __name__ == '__main__':

    mask = np.zeros((10,10), np.bool)
    mask[2:4, 2:4] = True
    mob = 2

    print(node_features(mask, mob))