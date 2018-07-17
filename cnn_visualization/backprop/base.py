#!/usr/bin/env python
# -*- coding:utf-8 -*-


from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn


class BackProp(metaclass=ABCMeta):
    """ abstract class for backpropagation object"""

    model: nn.Module
    gradients: torch.Tensor

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()

    def __call__(self,
                 input_image: torch.Tensor,
                 target_class: int
                 ) -> np.ndarray:

        return self.generate_gradients(input_image, target_class)

    @abstractmethod
    def generate_gradients(self,
                           input_image: torch.Tensor,  # Variableかも
                           target_class: int
                           ) -> np.ndarray:
        """ generate gradient from input, its class and inference result """

        pass
