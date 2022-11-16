"""
Module containing methods for outputing masks.

classes
-------
ConvMask
    Simple convolutional mask block. The sequence of operations is
        Pad -> Conv -> Nonlinear

Functions
---------
conv_mask(**kwargs)
    Interface to the ConvMask class that stores input arguments for use
    when the block is constructed at a later point.
"""

import torch
import torch.nn as nn
from dataops.utils import FunctionWrapper
from typing import Optional
from .padding import WrapPad2d
from ..utils import get_padding

def conv_mask(**kwargs):
    """
    Interface to the ConvMask class that stores input arguments for use
    when the block is constructed at a later point.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.mask import conv_mask
    >>> mask = conv_mask(nonlinear=torch.nn.Softplus())
    >>> mask(in_channels=8, out_channels=1)(torch.randn(1,8,16,16)).size()
    torch.Size([1, 1, 16, 16])
    """
    return FunctionWrapper(ConvMask, **kwargs)

class ConvMask(nn.Module):
    """
    Simple convolutional mask block. The sequence of operations is

        Pad -> Conv -> Nonlinear

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels.

    kernel_size : int, Tuple[int,int]
        The kernel size of the convolution

    dilation : int, Tuple[int,int]
        The dilation factor of the convolution

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True) or
        zero padding (False). Default is False.

    nonlinear : torch.nn.Module, optional.
        A nonlinear operation to apply. If set to `None`, then no operation
        is applied. Default is Sigmoid.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.mask import ConvMask
    >>> block = ConvMask(in_channels=3, out_channels=1, kernel_size=3)
    >>> block(torch.rand(1,3,16,16)).size()
    torch.Size([1, 1, 16, 16])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:Optional[nn.Module] = nn.Sigmoid(),
    ):
        super().__init__()

        pad   = (WrapPad2d if wrap_padding else nn.ZeroPad2d)(get_padding(kernel_size, dilation))
        conv  = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            dilation     = dilation,
        )

        block = [pad, conv]
        if nonlinear is not None:
            block.append(nonlinear)
        self.block = nn.Sequential(*block)

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input) 