import h5py
import numpy as np
from pathlib import Path
from skimage.morphology import binary_dilation
from typing import Tuple, Optional, Union
from pycocotools.mask import encode, decode, merge, toBbox


def img_from_output(
    path,
    enhance_edges=True,
):
    r"""
    Read SPPARKS meso input/output files and build an image where the candidate
    grain is centered in the matrix.

    Parameters
    ----------
    path: str or Path object
        path to directory containing initial.dream3d

    enhance_edges: bool
        if True, pixels identified to be on the grain boundary will be expanded
        through binary dilation to achieve more defined grain boundaries

    """
    path = Path(path)  # force path to Path object
    initfile = path / "initial.dream3d"

    assert initfile.is_file(), f"{str(initfile.absolute())} not found!"

    init = h5py.File(initfile, "r")
    sv = init["DataContainers"]["SyntheticVolume"]
    # grain_ids contains n x m array of pixels. Each pixel has integer value
    # indicating the grain id that it belongs to shifted by 1 so pixel ids start at 0

    grain_ids = np.asarray(sv["CellData"]["FeatureIds"]) - 1

    # grain_labels contains the class of each grain. The row index corresponds to the
    # grain in grain_ids (also shifted by 1 for consistency- the first row is zeros
    # to account for how the first grain id is 1- ie there is no 0 index, so this is
    # removed) In the original file there are 3 different vectors describing grain
    # mobility: [0.577, 0.577, 0.577] indicates high-mobility  (red in animation) [
    # 0.894, 0.447, 0] indicates low-mobility grain (blue in animation) [1, 0,
    # 0] indicates candidate grain (white in animation) Rather than keeping the
    # entire vector, we can get the label by simply counting the number of non-zero
    # elements and subtract 1 for zero-indexing 2 indicates high mobility,
    # 1 indicates low-mobility, 0 indicates candidate grain
    grain_labels = np.asarray(sv["CellFeatureData"]["AvgQuats"])[1:]
    grain_labels = (grain_labels > 0).sum(1) - 1

    # grain boundaries occur when a neighboring pixel has a different id
    mask_gb_c = grain_ids[:, :-1] != grain_ids[:, 1:]
    mask_gb_r = grain_ids[:-1, :] != grain_ids[1:, :]

    gids = grain_ids.copy()
    gids[:, :-1][mask_gb_c] = -1
    gids[:-1, :][mask_gb_r] = -1

    if enhance_edges:
        edges = binary_dilation(gids == -1, selem=np.ones((2, 2)))
    gids[edges] = -1

    # need temporary array to prevent collision between grain index id and mobility
    # label (can both be 0, 1, and 2)
    temp = gids.copy()
    for id, mob in enumerate(grain_labels):
        temp[gids == id] = mob
    gids = temp
    gids[gids == 0] = 3  # set candidate grain to highest value
    gids[gids == -1] = 0  # set grain boundary pixels to 0
    gids = gids / 3
    # now gids is a 'picture' of the grain structure
    # (ie each grain is separated by grain bonudaries) and each grain's intensity
    # is 0 for boundary, 1/3 for low mobility, 2/3 for high mobility, and 1 for
    # candidate grain. Ie grayscale image.

    gids = roll_img(gids, 1.0)  # center the grain in the image

    return gids


def roll_img(img, i):
    """
    Roll image *img* such that grain *i* is approximately centered.

    This is needed to account for periodic boundary conditions, which may cause an
    individual grain or grain's neighbor to be wrapped around an edge of the image.
    Wrapping to the center ensures that small grain neighborhoods will be centered
    and not cross the edges. Note that this only applies to small grain neighborhoods
    (ie the initial grain structure.) For grains that dominate the size of the image
    (ie after abnormal growth occurs,) then a more sophisticated approach (ie using
    np.where() to get coords and then operating on coords directly) will be needed,
    but this is likely not needed for determining the graph structure (ie initial
    grains are small,) and is not compatible with functions that operate on binary
    masks (ie skimage regionprops.)

    Parameters ---------- img: ndarray r x c integer numpy array where each element
    is the grain ID corresponding to the pixel

    i: int, float
        index of grain in *img* to center

    Returns
    -------
    img_roll: ndarray
        same format as *img* with coordinates rolled to center grain *i*
    """

    # center indices of image
    rmax, cmax = img.shape
    r, c = ((rmax + (rmax % 2)) // 2) - 1, ((cmax + (cmax % 2)) // 2) - 1

    # coordinates of grain of interest
    # needs to be sorted for all coords to appear in order
    rows, cols = (np.sort(x) for x in np.where(img == i))

    # TODO edge case: there exist disconnected grains of size 2 px
    #        ie rows (374, 376), cols (172, 170)
    #  this disconnection triggers logic for grain 'wrapping' around, even though it
    # does not. In fact, this is an edge case of SPPARKS itself where it incorrectly
    # labels two separated regions as the same grain

    # for both row and column indices:

    # if mask is only 1 pixel, roll it to center index directly

    # if mask has more than 1 pixels, look for discontinuities (difference in
    # coordinates between consecutive pixels in mask is larger than some threshold ie
    # 10px)

    # if no discontinuity is found (ie mask is not wrapped,) find position of mean
    # coordinate on each axis and roll so the mask is centered at the center of the
    # image

    # if a discontinuity is found: extend coordinates past the boundary of the image,
    # and then roll such that the mean position is at the center of the image

    if (
        len(rows) == 1
    ):  # mask is only 1 pixel, can't take difference in coords to detect split
        row_shift = r - rows[0]
    else:
        if (rows[1:] - rows[:-1]).max() > 1:  # grain wraps around top to bottom
            # verify that row actually spans image
            # there is an edge case where tiny grains can become
            # discontinuous despite not being wrapped
            if rows[0] == 0 and rows[-1] == rmax - 1:
                dr = rows[1:] - rows[:-1]
                rows[: dr.argmax() + 1] += rmax

        row_shift = r + rmax - int(rows.mean())

    if len(cols) == 1:  # mask is only 1 pixel
        col_shift = c - cols[0]
    else:
        if (cols[1:] - cols[:-1]).max() > 1:  # grain wraps from right to left
            if cols[0] == 0 and cols[-1] == cmax - 1:
                dc = cols[1:] - cols[:-1]
                cols[: dc.argmax() + 1] += cmax
        col_shift = c - int(cols.mean())

    img_roll = np.roll(img, (row_shift, col_shift), axis=(0, 1))

    return img_roll


def node_patch_to_image(rle: dict, mp: Tuple[int] = [0.8, 0.4, 0.6, 0.2]) -> np.ndarray:
    # maps 0 (candidate grain), 1 (blue grain), 2 (red grain) to values of 0.6, 0.2,
    # and 0.4, respectively adding 0.2 results in an image where background has
    # values 0.2, blue grain (lower mobility) has values 0.4, red grain (higher
    # mobility w candidate) has intensity 0.6, and candidate grain has intensity 0.8
    img = np.zeros(rle["size"], float) + mp[-1]
    img[decode(rle).astype(bool)] = mp[rle["type"]]

    return img


def edge_patch_to_image(
    patch: dict, edge_offset: float = 0.1, mp: Tuple[int] = (0.8, 0.4, 0.6, 0.2)
) -> np.ndarray:
    img = np.zeros(patch["size"], dtype=float) + mp[-1]
    for i in range(2):
        # fill in pixels for each grain with intensity corresponding to their type
        img[
            decode({"size": patch["size"], "counts": patch["counts"][i]}).astype(bool)
        ] = mp[patch["types"][i]]
        # highlight pixels on boundary. This way, the boundary is highlighted even if
        # the two connected grains have the same type
        img[patch["edge_coords"][i][0], patch["edge_coords"][i][1]] += edge_offset
    return img


def roll_shifts(img: np.ndarray, i: Optional[Union[int, float]] = None) -> Tuple[int]:
    """
    Compute shifts to pass to numpy.roll() to center an object in an image.

    Centering is determined by aligning the middle coordinate of the mask to the middle
     coordinate of the image, where the middle is defined as (i_max-i_min)//2.
     Centering can then be achieved with:
     shifts = roll_shifts(img, i)
     img_centered = numpy.roll(img, shifts, range(len(shifts)))

    Parameters
    ----------
    img: ndarray
        array to compute shifts for.
    i: optional numeric
        if None, img should be a binary integer image. Shifts for mask with value 1 will
            be computed.
        if specified, binary image will be determined by applying img == i, and then
            the shifts will be computed the samy way mentioned above

    Returns
    --------
    shifts: tuple(int)
        arguments passed to numpy.roll() to center the mask in the

    TODO example
    """

    if i is not None:
        img = img == i
    # coords of all pixels in img
    # sort so they are always exactly in order
    where = (np.sort(x) for x in np.where(img))

    shifts = [0 for _ in img.shape]
    for i, (seq, maxval) in enumerate(zip(where, img.shape)):
        # coordinates of center
        # for even numbered systems, picks the closest top left coord
        # ie for 2x2 image coord is 0, 0
        center = ((maxval + (maxval & 1)) >> 1) - 1
        if len(seq) == 1:
            # only 1 pixel: simply move it to center
            shifts[i] = center - seq[0]
        else:
            if seq[0] == 0 and seq[-1] == maxval - 1:  # grain wraps around image
                ds = seq[1:] - seq[:-1]
                # extend mask past upper limit of image to get mean coords
                seq[: ds.argmax() + 1] += maxval
                # above step breaks sorting of seq.
                # thus, min/max no longer at indices [0,-1]
            smin = seq.min()
            smax = seq.max()
            shifts[i] = center - (smin + ((smax - smin) // 2))
    return tuple(shifts)


def extract_node_patch(
    g, n: int, bounds: Tuple[int, int, int, int] = (208, 208, 304, 304)
) -> dict:
    """
    g: deepspparks graph Graph
        graph to extract node patch from
    n: int
        key of node in graph

    bounds: tuple(int)
        min row, min c, (inclusive) max row, max c (exclusive) used to select
        fixed-size image patch of grain
    """
    node = g.nodes[n]
    r1, c1, r2, c2 = bounds
    img = decode(node["rle"])
    shifts = roll_shifts(img)
    img = np.roll(img, shifts, range(len(shifts)))[r1:r2, c1:c2]
    rle = encode(np.asfortranarray(img))
    rle["type"] = node["grain_type"]
    return rle


# can't type hint g as Graph due to circular import
def extract_edge_patch(
    g, e: Tuple[int, int], box_size: int = 48, width: int = 2
) -> dict:
    # n1, n2: nodes in graph that rae connected by an edge
    # find union and roll it to the center of the image
    # get nodes from edge
    n1, n2 = (g.nodes[n] for n in e)

    boundary_thickness = width  # TODO refactor

    union = merge((n1["rle"], n2["rle"]), intersect=False)
    shift = roll_shifts(decode(union))
    union_decode = decode(union)
    union_decode_shift = np.roll(union_decode, shift, range(len(shift)))
    union_shift_encode = encode(union_decode_shift)
    bbox = toBbox(union_shift_encode)
    # convert from 1 corner + width + height to 4 corners
    bb = (
        int(bbox[0]),
        int(bbox[1]),
        int(bbox[0] + bbox[2] + 1),
        int(bbox[1] + bbox[3] + 1),
    )
    m1 = np.roll(decode(n1["rle"]), shift, range(len(shift)))[
        bb[1] : bb[3], bb[0] : bb[2]
    ]
    m2 = np.roll(decode(n2["rle"]), shift, range(len(shift)))[
        bb[1] : bb[3], bb[0] : bb[2]
    ]

    w1 = np.where(m1)
    w2 = np.where(m2)

    # shift from top left corner of full grain image to top left corner of current box
    total_shift = bbox[[1, 0]].astype(int)

    # find overlapping region  between masks for individual grains
    # --> boundary will be in overlap (+ width of at least 1 for perfectly
    # straigth, vertical/horizontal boundaries on rectangular grains)
    w = boundary_thickness  # width
    rmin = max((w1[0].min() - w, w2[0].min() - w, 0))
    rmax = min((w1[0].max() + w, w2[0].max() + w, m1.shape[0] - w)) + 1
    cmin = max((w1[1].min() - w, w2[1].min() - w, 0))
    cmax = min((w1[1].max() + w, w2[1].max() + w, m1.shape[1] - w)) + 1

    # coords of pixels within overlapping region
    gb_box_1 = m1[rmin:rmax, cmin:cmax]
    gb_box_2 = m2[rmin:rmax, cmin:cmax]

    # shift from top left corner to top left corner of overlapping region between grains
    total_shift += (rmin, cmin)

    w3 = np.where(gb_box_1)
    w4 = np.where(gb_box_2)

    # find pixels that are on boundary (city block distance 1 away from
    # pixels in other grain
    dist_threshold = boundary_thickness  # maximum city block distance to include on gb
    city_block = (
        np.abs(w3[0][:, np.newaxis] - w4[0][np.newaxis, :])
        + np.abs(w3[1][:, np.newaxis] - w4[1][np.newaxis, :])
    ) < dist_threshold + 1

    # Instead of doing city block distance for all pixels, it might be way faster to
    # look overlap with shifting coords or binary expansion
    # max of row or column i == 1 --> indices of coords of pixel
    # on boundary in w3 and w4
    cbm3 = city_block.max(1)
    cbm4 = city_block.max(0)

    final_offset = 0
    final_top = max(min(min(w3[0][cbm3]), min(w4[0][cbm4])) - final_offset, 0)
    final_bottom = (
        min(
            max(max(w3[0][cbm3]), max(w4[0][cbm4])) + final_offset,
            gb_box_1.shape[0] - 1,
        )
        + 1
    )

    final_left = max(min(min(w3[1][cbm3]), min(w4[1][cbm4])) - final_offset, 0)
    final_right = (
        min(
            max(max(w3[1][cbm3]), max(w4[1][cbm4])) + final_offset,
            gb_box_1.shape[1] - 1,
        )
        + 1
    )

    # shift from top left corner to top left corner of grain boundary
    # offset by 1 needed to exactly center the middle of the grain boundary
    # to the middle of the image. I think this is a result of toBbox returning
    # indices, not coordinates (ie the max index = max coordinate  + 1), but
    # am not 100% sure. I'm leaving it here since it was empirically found to work,
    # and this is a really complicated function. I reallly don't want to break anything.

    total_shift += (final_top - 1, final_left - 1)
    total_shift = (*total_shift, final_bottom - final_top, final_right - final_left)

    m1 = np.roll(decode(n1["rle"]), shift, range(len(shift)))
    m2 = np.roll(decode(n2["rle"]), shift, range(len(shift)))

    # generate index offsets such that x[center_row-offset0:center_row+offset1,
    #                                    center_col-offset2:center_col+offset2]
    # slices x to return a box_size x box_size slice
    # centered around center_row, center_col
    # for now, row and col offsets are the same because the box is assumed to be square
    # this can easily be changed later
    box_offsets = [
        ((box_size - 1) >> 1),
        ((box_size - 1) >> 1),
        (box_size >> 1) + 1,
        (box_size >> 1) + 1,
    ]

    center_gb_shift = np.array(
        [total_shift[0] + total_shift[2] // 2, total_shift[1] + total_shift[3] // 2]
    )

    # -final_top, -final_left shifts gb pixels to touch the top and left sides of image

    # shift coords of grain boundary to touch top and left sides of image (row/col 0),
    # and then shift the middle of the grain boundary to the center of the image
    where_shift = [
        -final_top + (box_size // 2) - ((final_bottom - final_top) // 2),
        -final_left + (box_size // 2) - ((final_right - final_left) // 2),
    ]

    # encode to rle with format:
    # {'size': total size of patch,
    # 'counts': [counts_mask_1, counts_mask_2],
    # 'types': [mask 1 grain type, mask 2 grain type]}
    # 'edge_coords': [[mask 1 row coords, mask 1 edge col coords],
    #                 [mask 2 edge row coords, mask 2 edge col coords]]}
    rle1 = encode(
        np.asfortranarray(
            m1[
                center_gb_shift[0]
                - box_offsets[0] : center_gb_shift[0]
                + box_offsets[2],
                center_gb_shift[1]
                - box_offsets[1] : center_gb_shift[1]
                + box_offsets[3],
            ]
        )
    )
    rle2 = encode(
        np.asfortranarray(
            m2[
                center_gb_shift[0]
                - box_offsets[0] : center_gb_shift[0]
                + box_offsets[2],
                center_gb_shift[1]
                - box_offsets[1] : center_gb_shift[1]
                + box_offsets[3],
            ]
        )
    )

    # row, col coords of pixels on each grain on the boundary
    edge_coords_all = [
        [y[cbm3] + o for y, o in zip(w3, where_shift)],
        [y[cbm4] + o for y, o in zip(w4, where_shift)],
    ]

    # unusually large boundaries may not fit in the box. Thus, we need to
    # exclude pixels that are outside the range of the final box
    # and would otherwise cause index errors.

    edge_coords = [
        [coords[coords < box_size] for coords in grain] for grain in edge_coords_all
    ]

    edge_patch = {
        "size": rle1["size"],
        "counts": [rle1["counts"], rle2["counts"]],
        "types": [n1["grain_type"], n2["grain_type"]],
        "edge_coords": edge_coords,
    }

    return edge_patch


if __name__ == "__main__":
    x = np.zeros((6, 6))
    x[1:4, 1:4] = 1
    z = x.copy()
    # roll and unroll image to verify that roll_img correctly centers it
    for rs in range(10):
        for cs in range(10):
            x = np.roll(x, (rs, cs), axis=(0, 1))
            y = roll_img(x, 1)
            assert np.all(y == z), f"{rs},{cs}"

    # TODO update unit test
    from deepspparks.graphs import Graph
    import matplotlib.pyplot as plt

    in_path = next(
        Path("../.unit_test/", "candidate-grains-toy-lmd-v2.0.0", "train").glob(
            "*.json"
        )
    )
    g = Graph.from_json(in_path)
    sg = g.get_subgraph(g.cidx[0], r=1)

    node_patch = node_patch_to_image(g.node_patch(sg.cidx[0]))
    e = [x for x in sg.eli if x[0] == sg.cidx[0]][1]
    edge_patch = edge_patch_to_image(g.edge_patch(e))
    fig, ax = plt.subplots(1, 2, dpi=300)
    ax[0].imshow(node_patch, vmin=0.2, vmax=0.9, cmap="magma")
    ax[0].set_title("node")
    ax[1].imshow(edge_patch, vmin=0.2, vmax=0.9, cmap="magma")
    ax[1].set_title("edge")
    fig.savefig("/host-debug/fig.png", bbox_inches="tight")
