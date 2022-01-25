import mlflow
from sklearn.metrics import confusion_matrix, classification_report


def cm(model, data, mask):
    """
    Simple wrapper to generate confusion matrix from model predictions

    Parameters
    ----------
    model: torch.nn.module
        model to evaluate
    data: torch_geometric.data.Data
        data to evaluate on
    mask: optional torch.Tensor or None
        If specified, boolean tensor that selects which nodes to include in predictions
        Otherwise, all nodes are included.

    Returns
    -------
    cm_: ndarray
        n_class x n_class confusion matrix
    """
    yp = model.predict(data)
    gt = data.y
    if mask is not None:
        yp = yp[mask]
        gt = gt[mask]
    cm_ = confusion_matrix(gt.numpy(), yp.numpy())
    return cm_


def log_classification_report(gt, yp, target_names, label):
    """
    Generates sklearn classification report and logs to mlflow tracking server.

    Parameters
    ----------
    gt: ndarray
        ground truth labels
    yp: ndarray
        predicted labels
    target_names: list(str)
        string ids corresponding to each int label in gt and yp
    label: str
        pre-pended before each label in classification report.
        For example, if label == 'train', then 'accuracy' key in
        classification report becomes 'train-accuracy'

    Returns
    -------
    None

    """

    cr = classification_report(gt, yp, target_names=target_names,
                               output_dict=True, zero_division=0)

    #  unpack metrics for each class (ie train-precision-NGG, val-precision-AGG)
    cr_final = {}
    for k, v in cr.items():
        if type(v) == dict:
            for vk, vv in v.items():
                cr_final['{}-{}-{}'.format(label, k, vk)] = vv
        else:
            cr_final['{}-{}'.format(label, k)] = v

    mlflow.log_metrics(cr_final)
    return
