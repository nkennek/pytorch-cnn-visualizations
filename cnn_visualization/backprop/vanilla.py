#!/usr/bin/env python
# -*- coding:utf-8 -*-


from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BackProp


class VanillaBackprop(BackProp):
    """
        Produces gradients generated with vanilla back propagation from the image
        https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/vanilla_backprop.py
    """

    def __init__(self, model: nn.Module,
                 device: torch.device = None) -> None:
        super(VanillaBackprop, self).__init__(model, device)
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self) -> None:
        def hook_function(module,
                          grad_in: Tuple[torch.Tensor],
                          grad_out: Tuple[torch.Tensor]):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer: nn.Module = list(self.model.children())[0]
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self,
                           input_image: torch.Tensor,
                           target_class: int) -> np.ndarray:

        # Forward
        model_output: torch.Tensor = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output: torch.Tensor = torch.FloatTensor(
            1, model_output.size()[-1]).to(self.device).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr: np.ndarray = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr
