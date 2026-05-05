"""
Temporal Convolutional Network (TCN) model for FOG detection.
"""

import torch
import torch.nn as nn
from pytorch_tcn.tcn import TCN


class TCNModel(nn.Module):
    """Causal TCN sequence model that outputs per-timestep FOG probabilities.

    Args:
        n_features: number of input channels (one per IMU axis)
    """

    def __init__(self, n_features):
        super().__init__()
        self.tcn = TCN(
            num_inputs=n_features,
            num_channels=[32, 64, 64, 32],
            kernel_size=5,
            dropout=0.1,
            causal=True,
            output_projection=1,
            output_activation=None,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inference=False):
        """
        Args:
            x:         (N, C, T) input tensor
            inference: if True, use TCN's causal buffer mode for streaming
                       (call model.tcn.reset_buffers() before a new trial)

        Returns:
            y: (N, T) FOG probability per timestep
        """
        return self.sigmoid(self.tcn(x, inference=inference)).squeeze(1)
