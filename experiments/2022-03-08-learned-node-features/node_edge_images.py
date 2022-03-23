from deepspparks.image import roll_img
import numpy as np
from pycocotools.mask import encode, decode, merge, toBbox


def extract_node_patch(n, bounds=(208, 208, 304, 304)):
    """
    n: dict
        node in graph. needs to contain fields 'rle' and 'grain_type'
    bounds: tuple(int)
        min row, min c, (inclusive) max row, max c (exclusive) used to select
        fixed-size image patch of grain
    """
    # n is a node: g.nodelist[n]

    # 0: candidate 1: blue 2: red

    mapper = [0.6, 0.2, 0.4]
    r1, c1, r2, c2 = bounds
    img = roll_img(decode(n["rle"]), 1)
    img = img[r1:r2, c1:c2] * mapper[n["grain_type"]] + 0.2
    return img


def roll_shifts(img, i=None):
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
    return shifts


def extract_edge_patch(g, e, box_size=48, width=2, edge_offset=0.1):
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

    mp = (0.8, 0.4, 0.6)
    x_img = np.zeros((box_size, box_size), float) + 0.2
    center_gb_shift = np.array(
        [total_shift[0] + total_shift[2] // 2, total_shift[1] + total_shift[3] // 2]
    )
    x_img[
        m1[
            center_gb_shift[0] - box_offsets[0] : center_gb_shift[0] + box_offsets[2],
            center_gb_shift[1] - box_offsets[1] : center_gb_shift[1] + box_offsets[3],
        ].astype(bool)
    ] = mp[n1["grain_type"]]
    x_img[
        m2[
            center_gb_shift[0] - box_offsets[0] : center_gb_shift[0] + box_offsets[2],
            center_gb_shift[1] - box_offsets[1] : center_gb_shift[1] + box_offsets[3],
        ].astype(bool)
    ] = mp[n2["grain_type"]]

    # -final_top, -final_left shifts gb pixels to touch the top and left sides of image

    # shift coords of grain boundary to touch top and left sides of image (row/col 0),
    # and then shift the middle of the grain boundary to the center of the image
    where_shift = [
        -final_top + (box_size // 2) - ((final_bottom - final_top) // 2),
        -final_left + (box_size // 2) - ((final_right - final_left) // 2),
    ]

    x_img[tuple(y[cbm3] + o for y, o in zip(w3, where_shift))] += edge_offset
    x_img[tuple(y[cbm4] + o for y, o in zip(w4, where_shift))] += edge_offset

    return x_img, final_bottom - final_top, final_right - final_left


if __name__ == "__main__":
    from deepspparks.graphs import Graph
    import matplotlib.pyplot as plt
    from pathlib import Path

    in_path = next(
        Path("../../.unit_test/", "candidate-grains-toy-lmd-v2.0.0", "train").glob(
            "*.json"
        )
    )
    g = Graph.from_json(in_path)
    sg = g.get_subgraph(g.cidx[0], r=1)

    node_patch = extract_node_patch(sg.nodes[sg.cidx[0]])
    edge_patch, dr, dc = extract_edge_patch(
        sg, [x for x in sg.eli if x[0] == sg.cidx[0]][1]
    )
    fig, ax = plt.subplots(1, 2, dpi=300)
    ax[0].imshow(node_patch, vmin=0.2, vmax=0.9, cmap="magma")
    ax[0].set_title("node patch")
    ax[1].imshow(edge_patch, vmin=0.2, vmax=0.9, cmap="magma")
    ax[1].set_title("edge patch ({}x{}px)".format(dr, dc))
    fig.savefig("/host-debug/fig.png", bbox_inches="tight")
