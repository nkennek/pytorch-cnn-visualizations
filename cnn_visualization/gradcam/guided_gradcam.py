#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import torch
import torch.nn as nn

from cnn_visualization.backprop.guidedbp import GuidedBackProp
from .gradcam import CamExtractor, GradCam


class GuidedGradCam(GradCam):
    """
        Guided Gradcam is just pointwise multiplication of cam mask and guided bp mask
    """

    model: nn.Module
    extractor: CamExtractor
    backprop: GuidedBackProp

    def __init__(self, model: nn.Module, target_layer: int,
                 device: torch.device,
                 scanner_args: dict = {},
                 verbose: bool = True) -> None:
        super(GuidedGradCam, self).__init__(
            model, target_layer, device, scanner_args, verbose)
        self.backprop = GuidedBackProp(model, device)

    def generate_cam(self,
                     input_image: torch.Tensor,
                     target_class: int
                     ) -> np.ndarray:

        gradcam_mask: np.ndarray = super(GuidedGradCam, self).generate_cam(
            input_image, target_class)
        guided_bp_mask: np.ndarray = self.backprop.generate_gradients(
            input_image, target_class)

        guided_cam_mask = np.multiply(gradcam_mask, guided_bp_mask)
        return guided_cam_mask
