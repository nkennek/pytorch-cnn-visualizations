#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import BackProp


class GuidedBackProp(BackProp):
    """
        Guided Back Propagation
        https://arxiv.org/abs/1412.6806
        https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py
    """

    def __init__(self, model: nn.Module) -> None:
        super(GuidedBackProp, self).__init__(model)
        self.update_relus()
        self.hook_layers()

    def hook_layers(self) -> None:
        def hook_function(module,
                          grad_in: Tuple[torch.Tensor],
                          grad_out: Tuple[torch.Tensor]):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self) -> None:
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module,
                               grad_in: Tuple[torch.Tensor],
                               grad_out: Tuple[torch.Tensor]):
            """ If there is a negative gradient, chenge it to zero """
            if isinstance(module, nn.ReLU):
                return torch.clamp(grad_in[0], min=0.0)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for _, module in self.model.features._modules.items():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self,
                           input_image: torch.Tensor,
                           target_class: int
                           ) -> np.ndarray:
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Conveert Pytorch variable to numpy array
        # [0] to get rid of the fiest channel (1, 3, 224, 224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
