"""
Module with recurrent structures.

Classes
-------
GRU
    Convolutional Gated Recurrent Unit (GRU) module.

LSTM
    Convolutional Long-Short Term Memory (LSTM) module.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from .padding import WrapPad2d
from ..utils import get_merged_channels, get_padding, merge_operations

class GRU(nn.Module):
    """
    Convolutional Gated Recurrent Unit (GRU) module.

    Attributes
    ----------
    gate1
        Convolutional operation for producing the outputs of the reset
        and update gates.
    
    gate2
        Convolutional operation for producing the output of the candidate
        vector gate.

    merge
        Callable function for merging the input tensor with the previous
        state.

    out_channels
        The number of output channels. Kept in memory for constructing
        a zero vector when `prev_state=None`

    Methods
    -------
    forward(input, prev_state=None)
        Returns the output of the cell based on the provided input tensor
        `input` and the previous state `prev_state`.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.recurrent import GRU
    >>> rnn = GRU(in_channels=3, out_channels=3, kernel_size=3, wrap_padding=False, merge_operation='add')
    >>> rnn(input=torch.randn(1,3,8,8), prev_state=None).size()
    torch.Size([1, 3, 8, 8])
    """
    def __init__(self, 
        in_channels:int,
        out_channels:int, 
        kernel_size:int = 3,
        dilation:int = 1,
        wrap_padding:bool = False, 
        merge_operation:str = 'concat',
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.

        out_channels : int
            The number of output channels.

        kernel_size : int
            The kernel size of the convolution.

        dilation : int
            The dilation factor of the convolution.

        wrap_padding : bool
            Boolean indicating whether to apply wrap padding (True) or
            zero padding (False). Default is False.

        merge_operation : Text
            String indicating the merge operation for combining the previous
            state with the input.

        **kwargs
            Additional parameters to pass into torch.nn.Conv2d

        Raises
        ------
        NotImplementedError
            If the provided merge operation is not implemented, then an
            exception is raised.
        
        ValueError
            If in_channels and out_channels are not compatible with the
            designated merge operation, then a ValueError is raised.
        """

        super().__init__()

        # ==========================================================
        # Make sure a valid merge operation is specified.
        # ==========================================================
        if merge_operation not in merge_operations.keys():
            raise NotImplementedError('Not a valid merge operation. Should be one of: {}'.format(merge_operations.keys()))
        self.merge = merge_operations[merge_operation]
        self.out_channels = out_channels

        # ==========================================================
        # Construct the gates. The first convolution represents the
        # output of two gates, which are split into their individual
        # components during the `forward` function.
        # ==========================================================
        Pad = (WrapPad2d if wrap_padding else nn.ZeroPad2d)(padding=get_padding(kernel_size, dilation))

        self.gate1 = nn.Sequential(
            Pad,
            nn.Conv2d(
                in_channels  = get_merged_channels(channels=[in_channels, out_channels], merge_operation=merge_operation),
                out_channels = 2 * out_channels,
                kernel_size  = kernel_size,
                dilation     = dilation,
                padding      = 0,
                **kwargs
            )
        )

        self.gate2 = nn.Sequential(
            Pad,
            nn.Conv2d(
                in_channels  = get_merged_channels(channels=[in_channels, out_channels], merge_operation=merge_operation),
                out_channels = out_channels,
                kernel_size  = kernel_size,
                dilation     = dilation,
                padding      = 0,
                **kwargs
            )
        )

    def forward(self, 
        input:torch.Tensor, 
        prev_state:Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns the output of the cell based on the provided input tensor
        `input` and the previous state `prev_state`.       

        Parameters
        ----------
        input : Tensor
            The input tensor.
        
        prev_state: Tensor, optional
            The previous state. If `prev_state=None` is set, then a
            zero Tensor is constructed and used.

        Returns
        -------
        hidden_state : Tensor
            The new hidden state
        """

        # ==========================================================
        # If no previous state is supplied (None), then generate
        # an empty state
        # ==========================================================
        if prev_state is None:
            batch_size = input.size(0)
            channels   = self.out_channels
            dimensions = input.size()[2:]
            size = [batch_size, channels, *dimensions]
            prev_state = torch.zeros(size, device=input.device)

        # ==========================================================
        # Calculate the reset and update gates.
        #   f_r = σ(W_r ⋅ [x, h_{t-1}])             [reset]
        #   f_u = σ(W_u ⋅ [x, h_{t-1}])             [update]
        # ==========================================================
        merged = self.merge(input, prev_state)
        gate_r, gate_u = torch.sigmoid(self.gate1(merged)).chunk(2,1)

        # ==========================================================
        # Calculate the candidate vector and new state
        #   f_h = tanh(W_h ⋅ [x, f_r × h_{t-1}]
        #   h_t = f_u⋅f_h + (1-f_u)⋅h_{t-1}
        # ==========================================================
        merged = self.merge(input, gate_r * prev_state)
        gate_h = torch.tanh(self.gate2(merged))

        return gate_u*gate_h + (1 - gate_u) * prev_state

class LSTM(nn.Module):
    """
    Convolutional Long-Short Term Memory (LSTM) module.

    Attributes
    ----------
    gates
        Convolutional operation for producing the outputs of each of the
        various gates.

    merge
        Callable function for merging the input tensor with the previous
        state.

    out_channels
        The number of output channels. Kept in memory for constructing
        a zero vector when `prev_state=None`

    Methods
    -------
    forward(input, prev_state=None)
        Returns the output of the cell based on the provided input tensor
        `input` and the previous state `prev_state`.

    Examples
    --------
    >>> import torch
    >>> from galnet.layers.recurrent import LSTM
    >>> rnn = LSTM(in_channels=3, out_channels=3, kernel_size=3, wrap_padding=False, merge_operation='add')
    >>> y = rnn(input=torch.randn(1,3,8,8), prev_state=None)
    >>> print(y[0].shape)
    torch.Size([1, 3, 8, 8])
    >>> print(y[1].shape)
    torch.Size([1, 3, 8, 8])
    """
    def __init__(self, 
        in_channels:int, 
        out_channels:int, 
        kernel_size:int = 3, 
        dilation:int = 1,
        wrap_padding:bool= False,
        merge_operation:str = 'concat',
        **kwargs
    ):
        """
        Parameters
        ----------
        in_channels : int
            The number of input channels.

        out_channels : int
            The number of output channels.

        kernel_size : int
            The kernel size of the convolution.

        dilation : int
            The dilation factor of the convolution.

        wrap_padding : bool
            Boolean indicating whether to apply wrap padding (True) or
            zero padding (False). Default is False.

        merge_operation : Text
            String indicating the merge operation for combining the previous
            state with the input.

        **kwargs
            Additional parameters to pass into torch.nn.Conv2d

        Raises
        ------
        NotImplementedError
            If the provided merge operation is not implemented, then an
            exception is raised.
        
        ValueError
            If in_channels and out_channels are not compatible with the
            designated merge operation, then a ValueError is raised.
        """
        super().__init__()

        # ==========================================================
        # Make sure a valid merge operation is specified.
        # ==========================================================
        if merge_operation not in merge_operations.keys():
            raise NotImplementedError('Not a valid merge operation. Should be one of: {}'.format(merge_operations.keys()))
        self.merge = merge_operations[merge_operation]
        self.out_channels = out_channels

        # ==========================================================
        # Construct the gate. The convolution represents the output
        # of four gates, which are split into their individual
        # components during the `forward` function.
        # ==========================================================
        Pad = WrapPad2d if wrap_padding else nn.ZeroPad2d

        self.gates = nn.Sequential(
            Pad(padding=get_padding(kernel_size, dilation)),
            nn.Conv2d(
                in_channels  = get_merged_channels(channels=[in_channels, out_channels], merge_operation=merge_operation),
                out_channels = 4 * out_channels,
                kernel_size  = kernel_size,
                dilation=dilation,
                padding=0,
                **kwargs
            )
        )

    def forward(self, 
        input:torch.Tensor, 
        prev_state:Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        detach:bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the output of the cell based on the provided input tensor
        `input` and the previous state `prev_state`.       

        Parameters
        ----------
        input : Tensor
            The input tensor.
        
        prev_state: (Tensor, Tensor), optional
            The previous state. If `prev_state=None` is set, then a
            zero Tensor is constructed and used.

        detach:bool
            Boolean indicating whether to detach the hidden state
            from the computational graph when applying the convolution.

        Returns
        -------
        hidden_state : Tensor
            The new hidden state

        cell_state : Tensor
            The new cell state
        """

        # ==========================================================
        # If no previous state is supplied (None), then generate
        # an empty state
        # ==========================================================
        if prev_state is None:
            batch_size = input.size(0)
            channels   = self.out_channels
            dimensions = input.size()[2:]
            size = [batch_size, channels, *dimensions]
            prev_state = (
                torch.zeros(size, device=input.device),
                torch.zeros(size, device=input.device)
            )

        # ==========================================================
        # Split previous state into hidden and cell components
        # ==========================================================
        hidden, cell = prev_state

        # ==========================================================
        # Join the input and hidden states
        # ==========================================================
        merged = self.merge(input, hidden.detach() if detach else hidden)

        # ==========================================================
        # Calculate the gate values. Equations are slightly different
        # from Equation 3 where the f and i gates have convolutions
        # with the previous cell state, but is more in line with
        # basic LSTMs.
        #   f: σ(W_f⋅[x,h] + b_f)           [forget]
        #   i: σ(W_i⋅[x,h] + b_i)           [input]
        #   o: σ(W_o⋅[x,h] + b_o)           [output]
        #   c: tanh(W_c⋅[x,h] + b_c)        [cell]
        # ==========================================================
        gate_f, gate_i, gate_o, gate_c  = self.gates(merged).chunk(4,1)
        gate_f = torch.sigmoid(gate_f)
        gate_i = torch.sigmoid(gate_i)
        gate_o = torch.sigmoid(gate_o)
        gate_c = torch.tanh(gate_c)

        # ==========================================================
        # Calculate the cell and hidden states
        #   Cₜ = fₜ ⋅ C_{t-1} + iₜ⋅cₜ
        #   Hₜ = oₜ ⋅ tanh(Cₜ)
        # ==========================================================
        cell   = (gate_f * cell) + (gate_i * gate_c)
        hidden = gate_o * torch.tanh(cell)

        return (hidden, cell)