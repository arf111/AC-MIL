import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0.001, path=['checkpoint.pt'], trace_func=print, score_name='dice', start_epoch=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (list): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.score_name = score_name
        self.start_epoch = start_epoch

    def __call__(self, val_score, model:list):

        current_score = val_score

        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(val_score, model, score_name=self.score_name)
        elif current_score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.save_checkpoint(val_score, model, score_name=self.score_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model:list, score_name='dice'):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation {score_name} decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        for i in range(len(self.path)):
            torch.save(model[i].state_dict(), self.path[i])

        self.val_loss_min = val_loss