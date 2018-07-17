#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Type

import numpy as np
import torch
import torch.nn as nn

from .base import BackProp
from .vanilla import VanillaBackprop


class SmoothGrad(BackProp):
    """
        SmoothGrad
        generate smoothed gradients out of backpropagation from image
        by adding random noise to input and average the gradient

        https://arxiv.org/abs/1706.03825
        https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/smooth_grad.py

    Args:
        backprop_model (VanillaBackprop): object to generate gradient
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """

    child_backprop: BackProp
    param_n: int
    param_sigma_multiplier: int
    target_class_num: int

    def __init__(self,
                 model: nn.Module,
                 param_n: int,
                 param_sigma_multiplier: int,
                 backpropCls: Type[BackProp] = VanillaBackprop
                 ) -> None:

        # super init is not called intentionally because this object utilizes
        # child BackProp object for main calculation.

        self.child_backprop = backpropCls(model)
        self.param_n = param_n
        self.param_sigma_multiplier = param_sigma_multiplier
        model_last_layer: nn.Module = list(
            self.child_backprop.model.children())[-1]
        self.target_class_num = model_last_layer.out_features

    def generate_gradients(self,
                           input_image: torch.Tensor,
                           target_class: int
                           ) -> np.ndarray:

        # Generate an empty image/matrix
        smooth_grad = np.zeros(input_image.size()[1:])

        mean = 0
        sigma = self.param_sigma_multiplier / \
            (torch.max(input_image) - torch.min(input_image)).data[0]
        for _ in range(self.param_n):
            # Generate noise

            noise = torch.Tensor(
                input_image.data.new(
                    input_image.size()
                ).normal_(mean, sigma**2))
            # Add noise to the image
            noisy_img = input_image + noise
            # Calculate gradients
            vanilla_grads = self.child_backprop.generate_gradients(
                noisy_img, target_class)
            # Add gradients to smooth_grad
            smooth_grad = smooth_grad + vanilla_grads
        # Average it out
        smooth_grad = smooth_grad / self.param_n
        return smooth_grad
