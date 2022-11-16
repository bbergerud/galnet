"""
Methods for applying padding to tensors.

Classes
-------
WrapPad2d(padding, dim)
    Implements wrap padding along the specified dimension.
    If wanting to apply wrap padding along both dimensions,
    check out Pytorch's implementation of circular padding.
    Wrapper to wrap_pad_2d.

Functions
---------
wrap_pad_2d(input, padding, dim, pad_both)
    Implements wrap padding along the specified dimension. Zero-padding
    is applied to the other dimension if `pad_both=True`. If wanting to
    apply wrap padding along both dimensions, check out Pytorch's
    implementation of circular padding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def wrap_pad_2d(
    input:torch.Tensor, 
    padding:int, 
    dim:int = 0,
    pad_both:bool = True,
) -> torch.Tensor:
    """
    Implements wrap padding along the specified dimension. Zero-padding
    is applied to the other dimension if `pad_both=True`. If wanting to
    apply wrap padding along both dimensions, check out Pytorch's
    implementation of circular padding.

    Parameters
    ----------
    input : Tensor
        The tensor to apply wrap padding to.

    padding : int
        The number of padding elements to apply to the top, right, left,
        and bottom. Designed to mimic torch.nn.ZeroPad2d.

    dim : int
        The dimension to apply the wrap padding along. Default is 0,
        which corresponds to the row. If `dim=1`, then the wrap padding
        occurs along the columns.

    pad_both : bool
        Boolean indicating whether to pad the non-wrapped axis with
        zeros. Default is True.

    Returns
    -------
    output : Tensor
        The padded tensor.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.padding import wrap_pad_2d
    >>> x = torch.arange(9).reshape(1,1,3,3)
    >>> print(wrap_pad_2d(x, padding=1, dim=0))
    tensor([[[[0, 6, 7, 8, 0],
              [0, 0, 1, 2, 0],
              [0, 3, 4, 5, 0],
              [0, 6, 7, 8, 0],
              [0, 0, 1, 2, 0]]]])
    >>> print(wrap_pad_2d(x, padding=1, dim=1, pad_both=False))
    tensor([[[[2, 0, 1, 2, 0],
              [5, 3, 4, 5, 3],
              [8, 6, 7, 8, 6]]]])
    """
    if dim == 0:
        pad1 = (0, 0, padding, padding)
        pad2 = (padding, padding, 0, 0)
    else:
        pad1 = (padding, padding, 0, 0)
        pad2 = (0, 0, padding, padding)

    input = F.pad(input, pad1, mode='circular')
    return F.pad(input, pad2, mode='constant') if pad_both else input


class WrapPad2d(nn.Module):
    """
    Implements wrap padding along the specified dimension.
    If wanting to apply wrap padding along both dimensions,
    check out Pytorch's implementation of circular padding.
    Wrapper to wrap_pad_2d.

    Parameters
    ----------
    padding : int
        The number of padding elements to apply to the top, right, left,
        and bottom. Designed to mimic torch.nn.ZeroPad2d.

    dim : int
        The dimension to apply the wrap padding along. Default is 0,
        which corresponds to the row. If `dim=1`, then the wrap padding
        occurs along the columns.

    Methods
    -------
    forward(x)
        Applies wrap padding to the tensor

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.padding import WrapPad2d
    >>> x = torch.arange(9).reshape(1,1,3,3)
    >>> y = WrapPad2d(padding=1,dim=0)(x)
    >>> print(y.squeeze())
    tensor([[0, 6, 7, 8, 0],
            [0, 0, 1, 2, 0],
            [0, 3, 4, 5, 0],
            [0, 6, 7, 8, 0],
            [0, 0, 1, 2, 0]])
    >>> y = WrapPad2d(padding=1, dim=1)(x)
    >>> print(y.squeeze())
    tensor([[0, 0, 0, 0, 0],
            [2, 0, 1, 2, 0],
            [5, 3, 4, 5, 3],
            [8, 6, 7, 8, 6],
            [0, 0, 0, 0, 0]])
    """
    def __init__(self, padding:int=1, dim:int=0) -> None:
        super().__init__()
        self.padding = padding
        self.dim = dim

    def forward(self, input:torch.Tensor, **kwargs) -> torch.Tensor:
        return wrap_pad_2d(input, padding=self.padding, dim=self.dim)