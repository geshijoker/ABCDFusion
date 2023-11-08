import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union

EPS = 1e-7

class DiceLoss(nn.Module):
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
    