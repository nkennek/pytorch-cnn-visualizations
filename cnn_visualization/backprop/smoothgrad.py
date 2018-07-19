#!/usr/bin/env python
# -*- coding:utf-8 -*-

from typing import Type

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

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
                 device: torch.device = None,
                 backpropCls: Type[BackProp] = VanillaBackprop
                 ) -> None:

        # super init is not called intentionally because this object utilizes
        # child BackProp object for main calculation.

        self.child_backprop = backpropCls(model, device)
        self.param_n = param_n
        self.param_sigma_multiplier = param_sigma_multiplier

    def generate_gradients(self,
                           input_image: Variable,
                           target_class: int
                           ) -> np.ndarray:

        # Generate an empty image/matrix
        smooth_grad = np.zeros(input_image.size()[1:])

        mean = 0
        sigma = self.param_sigma_multiplier / \
            (torch.max(input_image) - torch.min(input_image)).data[0]

        # prepare numpy image array to initialize leaf variable
        input_image_np = input_image.cpu().data.numpy()

        for _ in range(self.param_n):
            # Generate noise
            noise = np.random.normal(mean, sigma, size=input_image_np.shape)
            # Add noise to the image
            noisy_imp_np = input_image_np + noise
            noisy_img = torch.from_numpy(
                noisy_imp_np
            ).type(
                torch.FloatTensor
            ).to(
                self.child_backprop.device
            )
            noisy_img.requires_grad_()
            # Calculate gradients
            vanilla_grads = self.child_backprop.generate_gradients(
                noisy_img, target_class)
            # Add gradients to smooth_grad
            smooth_grad = smooth_grad + vanilla_grads

        # Average it out
        smooth_grad = smooth_grad / self.param_n
        return smooth_grad
