import matplotlib.pyplot as plt
import numpy as np

def pretty_cm(cm, labelnames, cscale=0.6, ax0=None, fs=6, cmap='cool'):
    """
    Generates a pretty-formated confusion matrix for convenient visualization.
    
    The true labels are displayed on the rows, and the predicted labels are displayed on the columns.
    
    Parameters
    ----------
    cm: ndarray 
        nxn array containing the data of the confusion matrix.
    
    labelnames: list(string)
        list of class names in order on which they appear in the confusion matrix. For example, the first
        element should contain the class corresponding to the first row and column of *cm*.
    cscale: float
        parameter that adjusts the color intensity. Allows color to be present for confusion matrices with few mistakes,
        and controlling the intensity for ones with many misclassifications.
    
    ax0: None or matplotlib axis object
        if None, a new figure and axis will be created and the visualization will be displayed.
        if an axis is supplied, the confusion matrix will be plotted on the axis in place.
    fs: int
        font size for text on confusion matrix.
        
    cmap: str
        matplotlib colormap to use
    
    Returns
    ---------
    None
    
    """
    
    acc = cm.trace() / cm.sum()
    if ax0 is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2.5, 2.5), dpi=300)
        fig.set_facecolor('w')
    else:
        ax = ax0

    n = len(labelnames)
    ax.imshow(np.power(cm, cscale), cmap=cmap, extent=(0, n, 0, n))
    labelticks = np.arange(n) + 0.5
    
    ax.set_xticks(labelticks, minor=True)
    ax.set_yticks(labelticks, minor=True)
    ax.set_xticklabels(['' for i in range(n)], minor=False, fontsize=fs)
    ax.set_yticklabels(['' for i in range(n)], minor=False, fontsize=fs)
    
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels=labelnames, minor=True, fontsize=fs)
    ax.set_yticklabels(labels=reversed(labelnames), minor=True, fontsize=fs)

    ax.set_xlabel('Predicted Labels', fontsize=fs)
    ax.set_ylabel('Actual Labels', fontsize=fs)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j + 0.5, n - i - 0.5, '{:^5}'.format(z), ha='center', va='center', fontsize=fs,
                bbox=dict(boxstyle='round', facecolor='w', edgecolor='0.3'))
    ax.grid(which='major', color=np.ones(3) * 0.33, linewidth=1)

    if ax0 is None:
        ax.set_title('Accuracy: {:.3f}'.format(cm.trace() / cm.sum()), fontsize=fs+2)
        plt.show()
        return
    else:
        return ax


def agg_cm(cmlist, fname):
    """
    Display confusion matrix for train, validation, and test performance and save to disk.
    
    Parameters
    ----------
    cmlist: list-like
        list-like structure containing confusion matrices in sklearn format
        (sklearn.metrics.confusion_matrix(y_true, y_predict)
         for the training, validation, and test sets respectively.
    
    fname: str or Path object
        Path to save figure.
    
    Returns
    ---------
    None
    
    Saves
    ---------
    cm_fig: image
        visualized confusion matrices saved to fname
    """
    
    fig, ax = plt.subplots(1,3, sharey=True, dpi=300, facecolor='w', figsize=(4,2))
    a = ax[0]
    a.set_yticks([0,1])
    a.set_yticklabels(['AGG', 'NGG'], fontsize=8)
    a.set_ylabel('True value', fontsize=8)
    for a, cm, title in zip(ax,cmlist,['Training','Validation','Test']):
        a.axis([-0.5,1.5,-0.5,1.5])
        a.set_aspect(1)
        #a.plot([0.5,0.5],[-0.5,1.5], '-k', linewidth=1)
        #a.plot([-0.5,1.5],[0.5,0.5], '-k', linewidth=1)
        a.set_xticks([0,1])
        a.set_xticks([0.5], minor=True)
        a.set_yticks([0.5], minor=True)
        a.set_xticklabels(['NGG','AGG'],fontsize=8)
        a.set_xlabel('Predicted Value', fontsize=8)
        a.set_title(title, fontsize=8)
        a.grid(which='minor', color='0')
        for (i, j), z in np.ndenumerate(cm):
            a.text(j, 1 - i, '{:^5}'.format(z), ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='w', edgecolor='0', linewidth=0.75))
    fig.tight_layout()
    fig.savefig(fname, bbox_inches='tight')
