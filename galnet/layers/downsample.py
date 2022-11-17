"""
Methods for downsampling tensors.

Class Methods
-------------
ConvDownsample
    Applies a strided convolution to reduce the spatial dimensions
    and change the number of channels. The sequence of operations is
        Pad -> Conv -> BN -> Nonlinear

ConvPoolDownsample
    Applies a pooling operation followed by a convolution to 
    reduce the spatial dimensions and change the number of
    channels. The sequence of operations is
        Pooling -> Pad -> Conv -> BN -> Nonlinear

PreactivationDownsample
    Applies a strided convolution to reduce the spatial dimensions
    and change the number of channels. The sequence of operations is
        BN -> Nonlinear -> Pad -> Conv

PreactivationPoolDownsample
    Applies a pooling operation followed by a convolution to 
    reduce the spatial dimensions and change the number of
    channels. The sequence of operations is
        Pooling -> BN -> Nonlinear -> Pad -> Conv

Pool2d
    Applies a pooling operation. If the input channels are not equal to the 
    output channels, then a 1x1 convolution is applied to match the channel sizes.
"""

import torch
import torch.nn as nn
from .blocks import ConvBlock, PreactivationBlock
from .padding import WrapPad2d
from ..utils import get_padding

class ConvDownsample(ConvBlock):
    """
    Applies a strided convolution to reduce the spatial dimensions
    and change the number of channels. The sequence of operations is

        Pad -> Conv -> BN -> Nonlinear

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels

    kernel_size : int
        The kernel size of the convolution
    
    stride : int
        The stride factor for downsampling

    dilation : int
        The dilation factor

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True)
        or zero padding (False).
    
    nonlinear : torch.nn.Module
        A nonlinear operation to apply

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.downsample import ConvDownsample as Downsample
    >>> x = torch.randn(1,1,16,16)
    >>> Downsample(in_channels=1, out_channels=1, stride=2)(x).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
    ):
        super().__init__(
            num_repeats=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            wrap_padding=wrap_padding,
            nonlinear=nonlinear,
        )

class ConvPoolDownsample(nn.Module):
    """
    Applies a pooling operation followed by a convolution to 
    reduce the spatial dimensions and change the number of
    channels. The sequence of operations is

        Pooling -> Pad -> Conv -> BN -> Nonlinear

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels

    kernel_size : int
        The kernel size of the convolution

    stride : int
        The stride factor for downsampling

    dilation : int
        The dilation factor

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True)
        or zero padding (False).
    
    nonlinear : torch.nn.Module
        A nonlinear operation to apply

    pooling : torch.nn.Module
        The pooling operation to apply

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.downsample import ConvPoolDownsample as Downsample
    >>> x = torch.randn(1,1,16,16)
    >>> Downsample(in_channels=1, out_channels=1, stride=2, pooling=torch.nn.MaxPool2d)(x).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int=2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
        pooling:nn.Module = nn.MaxPool2d,
    ):
        super().__init__()

        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d
        self.block = nn.Sequential(
            pooling(stride,stride),
            Pad(padding=get_padding(kernel_size, dilation)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nonlinear
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class PreactivationDownsample(PreactivationBlock):
    """
    Applies a strided convolution to reduce the spatial dimensions
    and change the number of channels. The sequence of operations is

        BN -> Nonlinear -> Pad -> Conv

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels

    kernel_size : int
        The kernel size of the convolution
    
    stride : int
        The stride factor for downsampling

    dilation : int
        The dilation factor

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True)
        or zero padding (False).
    
    nonlinear : torch.nn.Module
        A nonlinear operation to apply

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.downsample import PreactivationDownsample as Downsample
    >>> x = torch.randn(1,1,16,16)
    >>> Downsample(in_channels=1, out_channels=1, stride=2)(x).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int = 2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
    ):
        super().__init__(
            num_repeats=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            wrap_padding=wrap_padding,
            nonlinear=nonlinear,
        )

class PreactivationPoolDownsample(nn.Module):
    """
    Applies a pooling operation followed by a convolution to 
    reduce the spatial dimensions and change the number of
    channels. The sequence of operations is

        Pooling -> BN -> Nonlinear -> Pad -> Conv

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels

    kernel_size : int
        The kernel size of the convolution

    stride : int
        The stride factor for downsampling

    dilation : int
        The dilation factor

    wrap_padding : bool
        Boolean indicating whether to apply wrap padding (True)
        or zero padding (False).
    
    nonlinear : torch.nn.Module
        A nonlinear operation to apply

    pooling : torch.nn.Module
        The pooling operation to apply

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.downsample import PreactivationPoolDownsample as Downsample
    >>> x = torch.randn(1,1,16,16)
    >>> Downsample(in_channels=1, out_channels=1, stride=2, pooling=torch.nn.MaxPool2d)(x).size()
    torch.Size([1, 1, 8, 8])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        kernel_size:int = 3,
        stride:int=2,
        dilation:int = 1,
        wrap_padding:bool = False,
        nonlinear:nn.Module = nn.ReLU(),
        pooling:nn.Module = nn.MaxPool2d,
    ):
        super().__init__()

        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d
        self.block = nn.Sequential(
            pooling(stride,stride),
            nn.BatchNorm2d(in_channels),
            nonlinear,
            Pad(padding=get_padding(kernel_size, dilation)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=1,
            ),
        )

    def forward(self, input:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.block(input)

class Pool2d(nn.Module):
    """
    Applies a pooling operation. If the input channels are not equal to the 
    output channels, then a 1x1 convolution is applied to match the channel sizes.

    Parameters
    ----------
    in_channels : int
        The number of input channels
    
    out_channels : int
        The number of output channels

    stride : int
        The stride factor for downsampling

    pooling : torch.nn.Module
        The pooling operation to apply

    **kwargs
        Collects unused parameters to allow seamless interfacing
        with other downsampling methods.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.downsample import Pool2d as Downsample
    >>> x = torch.arange(16, dtype=torch.float32).reshape(1,1,4,4)
    >>> Downsample(in_channels=1, out_channels=1, stride=2, pooling=torch.nn.AvgPool2d)(x)
    tensor([[[[ 2.5000,  4.5000],
              [10.5000, 12.5000]]]])
    >>> Downsample(in_channels=1, out_channels=2, stride=2, pooling=torch.nn.MaxPool2d)(x).size()
    torch.Size([1, 2, 2, 2])
    """
    def __init__(self,
        in_channels:int,
        out_channels:int,
        stride:int=2,
        pooling:nn.Module = nn.MaxPool2d,
        **kwargs
    ):
        super().__init__()
        self.pool = pooling(stride, stride)

        if in_channels != out_channels:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, input:torch.Tensor, *args, **kwargs):
        output = self.pool(input)
        if 'conv' in self.__dict__['_modules']:
            output = self.conv(output)
        return output