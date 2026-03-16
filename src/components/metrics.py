import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.networks.utils import one_hot

class BraTSMetrics:
    def __init__(self):
        # We define metrics for the 3 clinical regions
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.hd95_metric = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", percentile=95)
        
        # Post-processing to convert model logits to discrete classes
        self.post_trans = AsDiscrete(argmax=True, to_onehot=4)
        self.label_trans = AsDiscrete(to_onehot=4)

    def compute_regions(self, y_pred, y):
        """
        Converts multi-class labels into BraTS clinical regions:
        WT (1+2+4), TC (1+4), ET (4)
        """
        # y_pred and y are expected to be one-hot [B, C, H, W, D]
        # Label 1: Core, 2: Edema, 3: (empty), 4: Enhancing
        
        y_pred_wt = torch.sum(y_pred[:, 1:, ...], dim=1, keepdim=True) > 0
        y_wt = torch.sum(y[:, 1:, ...], dim=1, keepdim=True) > 0
        
        y_pred_tc = torch.sum(y_pred[:, [1, 3], ...], dim=1, keepdim=True) > 0 # Indices 1 and 3 in one-hot correspond to labels 1 and 4
        y_tc = torch.sum(y[:, [1, 3], ...], dim=1, keepdim=True) > 0
        
        y_pred_et = y_pred[:, 3:4, ...] > 0
        y_et = y[:, 3:4, ...] > 0
        
        return (torch.cat([y_pred_wt, y_pred_tc, y_pred_et], dim=1), 
                torch.cat([y_wt, y_tc, y_et], dim=1))