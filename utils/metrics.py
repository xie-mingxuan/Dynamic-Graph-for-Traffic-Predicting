import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'average_precision': average_precision, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}


def get_link_prediction_metrics_multiclass(predicts: torch.Tensor, labels: torch.Tensor):
    """
    Get metrics for the multi-class link prediction task.
    :param predicts: Tensor, shape (num_samples, ), predicted class labels
    :param labels: Tensor, shape (num_samples, ), true class labels
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    # Convert tensors to numpy arrays
    predicts = predicts.cpu().detach().numpy()
    # Since the target indices are 1-indexed, we subtract 1 from them
    labels = labels.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(labels, predicts)
    precision = precision_score(labels, predicts, average='macro')
    recall = recall_score(labels, predicts, average='macro')
    f1 = f1_score(labels, predicts, average='macro')

    return {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1
    }
