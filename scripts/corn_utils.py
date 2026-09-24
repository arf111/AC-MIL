# Sebastian Raschka 2020-2021
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch


def corn_label_from_logits(logits):
    """
    Returns the predicted rank label from logits for a
    network trained via the CORN loss.

    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes)
        Torch tensor consisting of logits returned by the
        neural net.

    Returns
    ----------
    labels : torch.tensor, shape=(n_examples)
        Integer tensor containing the predicted rank (class) labels


    Examples
    ----------
    >>> # 2 training examples, 5 classes
    >>> logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
    ...                        [65.667, 0.303, 11.500, -4.524]])
    >>> corn_label_from_logits(logits)
    tensor([1, 3])
    """
    probas = torch.sigmoid(logits)
    
    if len(logits.shape) == 3:
        probas = torch.cumprod(probas, dim=2)
    else: 
        probas = torch.cumprod(probas, dim=1)
    
    predict_levels = probas > 0.5
    
    if len(logits.shape) == 3:
        predicted_labels = torch.sum(predict_levels, dim=2)
    else:
        predicted_labels = torch.sum(predict_levels, dim=1)
    
    return predicted_labels

def corn_probas(logits):
    probas = torch.sigmoid(logits)
    
    if len(logits.shape) == 3:
        probas = torch.cumprod(probas, dim=2)
    else: 
        probas = torch.cumprod(probas, dim=1)
    
    predict_levels = probas > 0.5
    
    if len(logits.shape) == 3:
        predicted_labels = torch.sum(predict_levels, dim=2)
    else:
        predicted_labels = torch.sum(predict_levels, dim=1)
    
    return predicted_labels, probas