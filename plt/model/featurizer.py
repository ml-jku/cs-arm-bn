import torch
from torch import nn
from plt.model.mlp import MLP
from plt.model.resnet import ResNet


class Featurizer(nn.Module):
    def __init__(self, featurizer, classifier, return_all_features=False):
        super().__init__()
        self.return_all_features = return_all_features
        self.featurizer = featurizer
        
        if hasattr(self.featurizer, "fc"):
            self.featurizer.fc = nn.Identity()

        self.classifier = nn.Linear(**classifier)


    def forward(self, x):
        features = self.featurizer(x)

        if isinstance(features, tuple):
            features, all_stats, all_pred_stats = features
        else:
            all_stats, all_pred_stats = None, None
        
        y_pred = self.classifier(features)

        if self.return_all_features:
            return features, y_pred
        else:
            # TODO: change
            if all_stats is not None:
                return y_pred, all_stats, all_pred_stats
            else:
                return y_pred