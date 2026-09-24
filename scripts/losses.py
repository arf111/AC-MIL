# Sebastian Raschka 2020-2021
# coral_pytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

import torch.nn.functional as F
import torch

def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.

    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1) or (num_examples, num_concepts, num_classes-1)
        Outputs of the CORN layer. Can handle both standard and concept-based predictions.

    y_train : torch.tensor, shape=(num_examples) or (num_examples, num_concepts)
        Torch tensor containing the class labels.

    num_classes : int
        Number of unique class labels (class labels should start at 0).

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.

    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    # Check if we have concept-based predictions (3D tensor)
    if len(logits.shape) == 3:
        # Shape: (bs, num_concepts, num_classes-1)
        bs, num_concepts, num_classes_minus_1 = logits.shape
        
        # Reshape to (bs * num_concepts, num_classes-1)
        logits_reshaped = logits.reshape(-1, num_classes_minus_1)
        
        # Reshape labels to (bs * num_concepts,)
        if len(y_train.shape) == 2:
            # Shape: (bs, num_concepts)
            y_train_reshaped = y_train.reshape(-1)
        else:
            # If y_train is 1D, replicate it for each concept
            y_train_reshaped = y_train.repeat_interleave(num_concepts)
        
        # Call the standard CORN loss with reshaped inputs
        sets = []
        for i in range(num_classes-1):
            label_mask = y_train_reshaped > i-1
            label_tensor = (y_train_reshaped[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits_reshaped[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                              + (F.logsigmoid(pred) - pred)*(1-train_labels))
            losses += loss

        return losses/num_examples
    
    else:
        # Standard 2D case: (num_examples, num_classes-1)
        sets = []
        for i in range(num_classes-1):
            label_mask = y_train > i-1
            label_tensor = (y_train[label_mask] > i).to(torch.int64)
            sets.append((label_mask, label_tensor))

        num_examples = 0
        losses = 0.
        for task_index, s in enumerate(sets):
            train_examples = s[0]
            train_labels = s[1]

            if len(train_labels) < 1:
                continue

            num_examples += len(train_labels)
            pred = logits[train_examples, task_index]

            loss = -torch.sum(F.logsigmoid(pred)*train_labels
                              + (F.logsigmoid(pred) - pred)*(1-train_labels))
            losses += loss

        return losses/num_examples
