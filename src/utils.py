import numpy as np

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
    return np.asarray(g.eli).T


if __name__ == "__main__":
    pass
