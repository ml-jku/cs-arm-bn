from torch import nn
from plt.model.mlp import MLP
from huggingface_mae import MAEModel


class OpenPhenom(nn.Module):
    def __init__(self, model_path, classifier, linear_probing=False):
        super().__init__()
        self.model = MAEModel.from_pretrained(model_path)

        if linear_probing:
            print("Freezing encoder")
            self.freeze_encoder()
        else:
            self.model.train()

        self.classifier = MLP(**classifier)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, X):
        features = self.model.predict(X)
        logits = self.classifier(features)
        return logits

    def featurizer(self, X):
        features = self.model.predict(X)
        return features