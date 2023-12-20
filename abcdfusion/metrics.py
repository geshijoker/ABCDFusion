from typing import List
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

EPS = 1e-7

def get_accuracy(cfm):
    """
    From 2x2 comfusion matrix to accuracy
    """
    tp = cfm[1][1]
    fp = cfm[0][1]
    fn = cfm[1][0]
    tn = cfm[0][0]
    return (tp+tn)/(tp+fp+fn+tn)

def get_precision(cfm):
    """
    From 2x2 comfusion matrix to precision
    """
    tp = cfm[1][1]
    fp = cfm[0][1]
    fn = cfm[1][0]
    tn = cfm[0][0]
    return tp/(tp+fp)

def get_recall(cfm):
    """
    From 2x2 comfusion matrix to recall
    """
    tp = cfm[1][1]
    fp = cfm[0][1]
    fn = cfm[1][0]
    tn = cfm[0][0]
    return tp/(tp+fn)

def get_f1(cfm):
    """
    From 2x2 comfusion matrix to f1 score
    """
    precision = get_precision(cfm)
    recall = get_recall(cfm)
    return 2*precision*recall/(precision+recall)

def get_confusion_matrix(preds, gtrue):
    """
    Compute confusion matrix from predictions and ground truth annotations
    Args: 
        preds: numpy array of predictions
        gtrue: numpy array of target labels
    Returns:
        2x2 confusion matrix
    """
    return confusion_matrix(gtrue, preds)

def get_group_confusion_matrix(preds, gtrue, groups):
    """
    Compute confusion matrix from predictions and ground truth annotations for each group given an array of groups
    Returns:
        A dictionary of confusion matrix for each group
    """
    cfm = {}
    group_set = set(groups)
    for group in group_set:
        mask = groups==group
        preds_group = preds[mask]
        gtrue_group = gtrue[mask]
        cfm[group] = confusion_matrix(gtrue_group, preds_group)
    return cfm
    
def count_support(cfm):
    """
    Count the number of instances to support the confusion matrix
    """
    return np.sum(cfm)

def compute_metric(cfm, metrics_list:List['str']=None):
    """
    Compute a list named metrics from a confusion matrix
    Args:
        cfm: 2x2 confusion matrix
        metrics_list: a list of predefined metric
    Returns: 
        A dictionary of (metric name, values computed from confusion matrix)
    """
    ans = {}
    for metric_name in metrics_list:
        if metric_name == 'accuracy':
            ans[metric_name] = get_accuracy(cfm)
        elif metric_name == 'precision':
            ans[metric_name] = get_precision(cfm)
        elif metric_name == 'recall':
            ans[metric_name] = get_recall(cfm)
        else:
            sys.exit('{} is not supported'.format(metric_name))
    return ans
    
def compute_metrics(cfm_dict, metrics_list:List['str']=None):
    """
    Compute a list named metrics from a confusion matrices of groups
    Args:
        cfm_dict: a dictionary of (group, confusion matrix)
        metrics_list: a list of predefined metric
    Returns: 
        A nested dictionary of (number of instances in the group, (group, metrics values))
    """
    ans = {}
    for group, cfm in cfm_dict.items():
        ans[group] = {'support': int(count_support(cfm))}
        ans[group].update(compute_metric(cfm, metrics_list))
    return ans

class DiceLoss(nn.Module):
    """
    Dice loss -- https://paperswithcode.com/method/dice-loss
    """
    def __init__(self, softmax=False):
        super(DiceLoss, self).__init__()
        self.softmax = softmax
        self.smooth = EPS

    def forward(self, pred, target):
        if self.softmax:
            pred = F.softmax(pred,1)
        pred_flat = pred.contiguous().view(-1)
        true_flat = target.contiguous().view(-1)
        intersection = (pred_flat * true_flat).sum()
        union = torch.sum(pred_flat) + torch.sum(true_flat)
        
        return 1 - ((2. * intersection + self.smooth) / (union + self.smooth) )
    
class WeightedBCELoss(nn.Module):
    """Computes the BCEloss weighted by manual class weights
       Args:
           weights (FloatTensor): The float tensor to define weights
    """
    def __init__(self, weights, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        if torch.is_tensor(weights):
            self.weights = weights
        else:
            self.weights = torch.Tensor(weights)
        assert len(self.weights) == 2, "The weights dimension is not 2!"
        self.weights.requires_grad = False
        self.reduction = reduction
        
    def forward(self, pred, target):
        bceloss = nn.BCELoss(reduction='none')
        loss = bceloss(pred, target)
        loss *= self.weights.to(loss.device)
        if self.reduction=='mean':
            return loss.mean()
        elif self.reduction=='sum':
            return loss.sum()
        else:
            return loss

class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2
    Focal loss applies a modulating term to the cross entropy loss in order to focus learning on hard misclassified examples. It is a dynamically scaled cross entropy loss, where the scaling factor decays to zero as confidence in the correct class increases. 

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """
    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
            ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
                )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0], device=target.device)
        weights = target * self.weights.to(target.device)
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
            ) -> Tensor:
        
        #convert all ignore_index elements to zero to avoid error in one_hot
        #note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target!=self.ignore_index) 
        target = target.view(-1).long()
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
            ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        # mask = mask.view(-1)
        # x = self._process_preds(x)
        num_classes = x.shape[-1]
        # target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        print(target.shape, x.shape, mask.shape)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x
        
        
if __name__=='__main__':

    # The weights parameter is similar to the alpha value mentioned in the paper
    weights = torch.FloatTensor([0.3, 0.7])
    criterion = FocalLoss(gamma=0.7, weights=weights)
    
    batch_size = 10
    m = torch.nn.Sigmoid()
    logits = torch.randn(batch_size)
    target = torch.randint(0, 2, size=(batch_size,))
    loss = criterion(m(logits), target)
    print(loss)
    