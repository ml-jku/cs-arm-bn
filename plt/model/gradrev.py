import torch
from torch import nn
from torch.autograd import Function
from typing import Tuple, Any, Optional

from hydra.utils import instantiate

from plt.model.mlp import MLP
from plt.model.resnet import ResNet


class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None



class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class GradRev(nn.Module):
    def __init__(self, featurizer, classifier, domain_classifier):
        super().__init__()

        self.featurizer = featurizer
        self.classifier = MLP(**classifier)
        self.domain_classifier = MLP(**domain_classifier)
        self.gradient_reverse_layer = GradientReverseLayer()
 

    def forward(self, x):
        features = self.featurizer(x)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(features)
        return y_pred, domains_pred

