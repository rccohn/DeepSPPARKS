import matplotlib.pyplot as plt
import mlflow
import numpy as np


def pretty_cm(cmlist, class_labels, return_figure=True, fpath=None, artifact_path=None):
    """
    Display confusion matrix for train, validation, and test performance and save to disk.

    Parameters
    ----------
    cmlist: list-like, or dict
        list-like structure containing confusion matrices in sklearn format
        (sklearn.metrics.confusion_matrix(y_true, y_predict)
         for the training, validation, and [optional test] sets respectively.
         if dict, keys should be ('train','val','test') and values should
         be the corresponding confusion matrix

    class_labels: list-like container of strings:
        string labels corresponding to each index in the confusion matrix

    return_figure: Bool
        if True, figure is returned by function

    fpath: None or str or Path object
        if None, figure is not saved to disk or logged as artifact
        Otherwise, Path to save figure to.

    artifact_path: str or None
        if None, figure is not logged as artifact
        else:
         - fpath must not be none
         - if str, path in artifact directory to log artifact to with mlflow.log_artifact

    Returns
    ---------
    fig: matplotlib Figure or None
        if return_figure == True, figure is returned
        Otherwise, None is returned.

    Saves
    ---------
    cm_fig: image
        visualized confusion matrices saved to fname
    """
    if type(cmlist) == dict:
        cmlist = [cmlist['train'], cmlist['val'], cmlist.get('test', None)]
        cmlist = [x for x in cmlist if x is not None]  # remove test if it is not in dict
    subsets = ['Train', 'Valid', 'Test'][:len(cmlist)]
    fig, ax = plt.subplots(1, len(cmlist), sharey=True, dpi=300,
                           facecolor='w', figsize=(4 / 3 * len(cmlist), 2))
    n = len(class_labels)
    major_ticks = np.arange(n+1)
    minor_ticks = np.arange(n) + 0.5

    a = ax[0]
    a.set_yticks(major_ticks)
    a.set_yticks(minor_ticks, minor=True)
    a.set_yticklabels(class_labels[::-1], fontsize=8, minor=True)
    a.set_yticklabels(['' for _ in a.get_yticks()], minor=False)

    a.set_ylabel('True value', fontsize=8)

    for a, cm, title in zip(ax, cmlist, subsets):
        a.set_aspect(1)
        # a.plot([0.5,0.5],[-0.5,1.5], '-k', linewidth=1)
        # a.plot([-0.5,1.5],[0.5,0.5], '-k', linewidth=1)
        a.set_xticks(major_ticks)
        a.set_xticks(minor_ticks, minor=True)
        a.set_xticklabels(class_labels, fontsize=8, minor=True)
        a.set_xticklabels(['' for _ in a.get_yticks()], minor=False)
        a.set_title(title, fontsize=8)
        a.grid(which='major', color='0')
        for (i, j), z in np.ndenumerate(cm):
            a.text(j+0.5, n-0.5 - i, '{:^5}'.format(z), ha='center', va='center', fontsize=8,
                   bbox=None)

    # fig.tight_layout()
    fig.text(0.4, 0.1, 'Predicted value', fontsize=8)
    if fpath is not None:
        fig.savefig(fpath, bbox_inches='tight')
        if artifact_path is not None:
            mlflow.log_artifact(str(fpath.absolute()), artifact_path)

    if return_figure:
        return fig


if __name__ == "__main__":
    # simple test case for verification
    pretty_cm(([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]],
               [[10, 11, 12],
                [13, 14, 15],
                [16, 17, 18]]
               ),
              ['a', 'b', 'c'])
    plt.show()
