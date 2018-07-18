#!/usr/bin/env python
# -*- coding:utf-8 -*-


from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class CamExtractor(object):
    """ Extractor for GradCam"""

    model: nn.Module
    target_layer: int
    gradients: torch.Tensor

    def __init__(self, model: nn.Module, target_layer: int) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad: torch.Tensor) -> None:
        """ save backward gradient to member variable """
        self.gradients = grad

    def forward_pass_on_convolutions(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer

        return conv_output, x

    def forward_pass(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam(object):
    """
        Grad-Cam
        Produces class activation map
        https://arxiv.org/abs/1610.02391
        https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/gradcam.py
    """

    model: nn.Module
    extractor: CamExtractor

    def __init__(self, model: nn.Module, target_layer: int) -> None:
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self,
                     input_image: torch.Tensor,
                     target_class: int = None
                     ) -> np.ndarray:
        """ generate gradcam heatmap """

        # Full forward pass
        # conv_output is the output of convolutions as specified layer
        # model_output is the final output of the model (1, num_classes)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            # set argmax to target class in case not specified
            target_class = np.argmax(model_output.data.numpy())

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]

        # Get convolution outputs
        target = conv_output.data.numpy()[0]

        # Get weights from gradients by taking averages
        weights = np.mean(guided_gradients, axis=(1, 2))

        # Create empty numpy array for cam
        cam: np.ndarray = np.ones(target.shape[1:], dtype=np.float32)

        # Multiply each weights with its conv output and sum
        for i, weight in enumerate(weights):
            cam += weight * target[i, :, :]

        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0)
        cam = (cam - np.min * cam) - (np.max(cam) -
                                      np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        return cam
