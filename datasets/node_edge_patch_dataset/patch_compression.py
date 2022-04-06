import numpy as np


def compress_node_patch(patch: dict) -> str:
    """
    Compress node patch representation, removing redundant
    information, and converting it to a string
    that can efficiently be saved to a text file.
    """
    return "".join((str(patch["type"]), patch["counts"].decode("utf-8")))


def recover_node_patch(patch: str, size: int) -> dict:
    """
    reverse compress_node_patch for verifying that it
    works correctly
    """
    return {
        "type": int(patch[0]),
        "size": [size, size],
        "counts": patch[1:].encode("utf-8"),
    }


def compress_edge_patch(patch: dict) -> str:
    """
    Compress edge patch representation, remove redundant
    information, and convert to a string that
    can efficiently be saved to a text file
    """
    # first line: first character indicates grain ids of both
    # grains on the boundary, and the rle counts for the first grain
    patch_compress = []
    patch_compress.append(
        "".join(
            (
                str(3 * patch["types"][0] + patch["types"][1]),
                patch["counts"][0].decode("utf-8"),
            )
        )
    )
    # second line: rle counts of second grain
    patch_compress.append(patch["counts"][1].decode("utf-8"))
    # third and fourth lines: row, col coordinates, of boundary
    # pixels on first and second grain.
    # Each coordinate is saved as a 0-padded 2 digit number
    # (almost all are 2 digit numbers, so trying to distinguish
    # 1 digit coordinates eliminates all space saved from
    # getting rid of trailing 0)
    # the firt half of the sequence is row coordinates, and
    # the second half of the sequence is col coordinates

    patch_compress.append(
        "".join(["{:02d}".format(y) for z in patch["edge_coords"][0] for y in z])
    )
    patch_compress.append(
        "".join(["{:02d}".format(y) for z in patch["edge_coords"][1] for y in z])
    )
    return "\n".join(patch_compress)


def recover_edge_patch(patch_c: str, size: int) -> dict:
    """
    reverse compress_edge_patch for verifying that it
    works correctly
    """
    patch_c = patch_c.split("\n")
    patch = {}
    patch["size"] = [size, size]
    patch["counts"] = [patch_c[0][1:].encode("utf-8"), patch_c[1].encode("utf-8")]
    patch["types"] = [int(patch_c[0][0]) // 3, int(patch_c[0][0]) % 3]
    s = patch_c[2]
    coords_1 = np.array([int("".join(x)) for x in list(zip(s[::2], s[1::2]))])
    n1 = len(coords_1) // 2
    s = patch_c[3]
    coords_2 = np.array([int("".join(x)) for x in list(zip(s[::2], s[1::2]))])
    n2 = len(coords_2) // 2
    patch["edge_coords"] = [
        [np.array(coords_1[:n1]), np.array(coords_1[n1:])],
        [np.array(coords_2[:n2]), np.array(coords_2[n2:])],
    ]
    return patch
