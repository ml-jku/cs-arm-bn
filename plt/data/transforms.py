import torch
import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from torchvision.transforms import functional as F
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from torchvision.transforms import ToTensor


class TVN():
    """
    Typical Variation Normalization. Implemented as here https://github.com/recursionpharma/EFAAR_benchmarking/.
    """
    def __init__(self, pcafeat=True):
        self.pcafeat = pcafeat

        if self.pcafeat:
            self.pca = PCA()

        self.scaler = StandardScaler()

    def fit(self, X_neg, batch_neg): 
        X_neg = self.scaler.fit_transform(X_neg)
        X_neg = self.pca.fit_transform(X_neg)

        X_neg_transformed = np.empty_like(X_neg)

        for b in set(batch_neg):
            pca_scaler = StandardScaler()
            neg_indices = np.where(np.array(batch_neg) == b)
            X_neg_b = X_neg[neg_indices]
            X_neg_b = pca_scaler.fit_transform(X_neg_b)
            X_neg_transformed[neg_indices] = X_neg_b

        self.target_cov = self.get_cov(X_neg_transformed)

        
    def transform(self, X_pos, batch_pos, X_neg, batch_neg):
        common_batches = set(batch_pos).intersection(set(batch_neg))

        X_pos_transformed = np.empty_like(X_pos)

        for b in common_batches:
            pca_scaler = StandardScaler()
            
            neg_indices = np.where(np.array(batch_neg) == b)
            X_neg_b = X_neg[neg_indices]

            X_neg_b = self.scaler.transform(X_neg_b)
            X_neg_b = self.pca.transform(X_neg_b) if self.pcafeat else X_neg_b
            X_neg_b = pca_scaler.fit_transform(X_neg_b)
            source_cov = self.get_cov(X_neg_b)

            pos_indices = np.where(np.array(batch_pos) == b)

            X_pos_b = X_pos[pos_indices]

            X_pos_b = self.scaler.transform(X_pos_b)
            X_pos_b = self.pca.transform(X_pos_b)
            X_pos_b = pca_scaler.transform(X_pos_b)

            X_pos_b = np.matmul(X_pos_b, linalg.fractional_matrix_power(source_cov, -0.5))
            X_pos_b = np.matmul(X_pos_b, linalg.fractional_matrix_power(self.target_cov, 0.5))

            X_pos_transformed[pos_indices] = X_pos_b

        return X_pos_transformed


    def get_cov(self, X):
        return np.cov(X, rowvar=False, ddof=1) + 0.5 * np.eye(X.shape[1])



def _transform(n_px_tr: int, n_px_val: int, is_train: bool, normalize:str = "dataset", preprocess: str = "downsize", stats:str = None, transf=None):
    preprocess = preprocess.split("+")

    if ("crop" in preprocess) and ("downsize" in preprocess):
        print("Crop and downsize are mutually exclusive. Using downsize")
    
    transforms = []

    if is_train:
        if "crop" in preprocess:
            transforms.append(v2.RandomCrop(n_px_tr))
        if "downsize" in preprocess:
            transforms.append(v2.RandomResizedCrop(n_px_tr, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC))
        if "randomresized" in preprocess:
            transforms.append(v2.RandomResizedCrop(n_px_tr, scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC))
        if "rotate" in preprocess:
            transforms.append(v2.RandomRotation(180))
        if "flip" in preprocess:
            transforms.extend([v2.RandomHorizontalFlip(), v2.RandomVerticalFlip()])
        if "gaussianblur" in preprocess:
            transforms.append(v2.GaussianBlur(kernel_size=(5,5)))
        if "colorjitter" in preprocess:
            transforms.append(v2.ColorJitter(brightness=0.0, contrast=1.0, saturation=0.0, hue=0.0))
        if "equalize" in preprocess:
            transforms.append(v2.RandomEqualize())    
                
    else:
        if "crop" in preprocess or "randomresized" in preprocess:
            transforms.append(v2.CenterCrop(n_px_val))
            
        elif "downsize" in preprocess:
            transforms.extend([
                        v2.Resize(n_px_val, interpolation=InterpolationMode.BICUBIC),
                        v2.CenterCrop(n_px_val),
                      ])

    transforms.append(v2.ToDtype(torch.float32, scale=True)) 

    if normalize:
        if normalize == "img":
            normalize = NormalizeByImage()
        elif normalize == "dataset":
            mean, std = stats['mean'], stats['std']
            normalize = v2.Normalize(mean, std)  
        elif normalize == "batch":
            normalize = ConditionalNormalize(stats)
            
        if not isinstance(normalize, ConditionalNormalize):
            transforms.append(normalize)

    print(transforms)
    print(normalize)

    if isinstance(normalize, ConditionalNormalize):
        final_transforms = [v2.Compose(transforms), normalize]
    else:
        final_transforms = [v2.Compose(transforms)]
    
    print(final_transforms)

    return final_transforms




class ConditionalNormalize(torch.nn.Module):
    """Normalize a tensor image conditioned on a certain label.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, stats: dict, inplace=False):
        super().__init__()
        self.stats = stats
        self.inplace = inplace

    def forward(self, tensor, y):
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.
            y (int): label for metadata

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean, std = self.stats[y]["mean"], self.stats[y]["std"]

        return F.normalize(tensor, mean, std, self.inplace)


class NormalizeByImage(torch.nn.Module):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        # for t in tensor:
        #     t.sub_(t.mean()).div_(t.std() + 1e-7)
        mean = tensor.mean(dim=(1, 2))
        std = tensor.std(dim=(1, 2))
        std[std == 0.] = 1.

        return F.normalize(tensor, mean, std)
    

