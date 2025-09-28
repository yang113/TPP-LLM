"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class msle_loss(nn.Module):
    def __init__(self):
        super(msle_loss, self).__init__()

    def forward(self,forecast: t.Tensor, target: t.Tensor) -> t.float:
        return t.mean(t.square(t.log2(forecast+1) - t.log2(target+1)))

