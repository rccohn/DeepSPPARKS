import h5py
import numpy as np
from pathlib import Path
from skimage.morphology import binary_dilation


def img_from_output(path, enhance_edges=True, ):
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
    initfile = path / 'initial.dream3d'


    assert initfile.is_file(), f'{str(initfile.absolute())} not found!'

    init = h5py.File(initfile, 'r')
    sv = init['DataContainers']['SyntheticVolume']
    # grain_ids contains n x m array of pixels. Each pixel has integer value indicating the grain id that it belongs to
    # shifted by 1 so pixel ids start at 0

    grain_ids = np.asarray(sv['CellData']['FeatureIds']) - 1

    # grain_labels contains the class of each grain.
    # The row index corresponds to the grain in grain_ids (also shifted by 1 for consistency- the first row is zeros
    # to account for how the first grain id is 1- ie there is no 0 index, so this is removed)
    # In the original file there are 3 different vectors describing grain mobility:
    # [0.577, 0.577, 0.577] indicates high-mobility  (red in animation)
    # [0.894, 0.447, 0] indicates low-mobility grain (blue in animation)
    # [1, 0, 0] indicates candidate grain (white in animation)
    # Rather than keeping the entire vector, we can get the label by simply counting the number of non-zero elements
    #  and subtract 1 for zero-indexing
    # 2 indicates high mobility, 1 indicates low-mobility, 0 indicates candidate grain
    grain_labels = np.asarray(sv['CellFeatureData']['AvgQuats'])[1:]
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

    This is needed to account for periodic boundary conditions, which may cause an individual grain
    or grain's neighbor to be wrapped around an edge of the image. Wrapping to the center ensures that small grain
    neighborhoods will be centered and not cross the edges. Note that this only applies to small grain neighborhoods
    (ie the initial grain structure.) For grains that dominate the size of the image (ie after abnormal growth occurs,)
    then a more sophisticated approach (ie using np.where() to get coords and then operating on coords directly) will
    be needed, but this is likely not needed for determining the graph structure (ie initial grains are small,) and is
    not compatible with functions that operate on binary masks (ie skimage regionprops.)

    Parameters
    ----------
    img: ndarray
        r x c integer numpy array where each element is the grain ID corresponding to the pixel

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

    # for both row and column indices:

    # if mask is only 1 pixel, roll it to center index directly

    # if mask has more than 1 pixels, look for discontinuities (difference in coordinates between consecutive
    # pixels in mask is larger than some threshold ie 10px)

    # if no discontinuity is found (ie mask is not wrapped,) find position of mean coordinate on each axis
    # and roll so the mask is centered at the center of the image

    # if a discontinuity is found: extend coordinates past the boundary of the image, and then roll
    # such that the mean position is at the center of the image

    if len(rows) == 1:  # mask is only 1 pixel, can't take difference in coords to detect split
        row_shift = (r - rows[0])
    else:
        if (rows[1:] - rows[:-1]).max() > 1:  # grain wraps around top to bottom
            dr = rows[1:]-rows[:-1]
            rows[:dr.argmax()+1] += rmax
        row_shift = (r - int(rows.mean()))

    if len(cols) == 1:  # mask is only 1 pixel
        col_shift = (c - cols[0])
    else:
        if (cols[1:] - cols[:-1]).max() > 1:  # grain wraps from right to left
            dc = cols[1:]-cols[:-1]
            cols[:dc.argmax()+1] += cmax
        col_shift = (c - int(cols.mean()))

    img_roll = np.roll(img, (row_shift, col_shift), axis=(0, 1))

    return img_roll


if __name__ == "__main__":
    x = np.zeros((6,6))
    x[1:4, 1:4] = 1
    z = x.copy()
    for rs in range(10):
        for cs in range(10):
            x = np.roll(x, (rs, cs), axis=(0, 1))
            y = roll_img(x, 1)
            assert np.all(y == z), f'{rs},{cs}'