# Code from https://github.com/DequanWang/tent/blob/master/tent.py
import torch 
from hydra.utils import instantiate
from copy import deepcopy
from plt.trainer.ERM import ERM

import PIL
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string


def get_tta_transforms(img_size, gaussian_std: float=0.005, soft=False, padding_mode='edge', cotta_augs=True):
    n_pixels = img_size[0] if isinstance(img_size, (list, tuple)) else img_size

    tta_transforms = [
        # We removed color jittering as these augmentations do not work on normalized imgs.
        # ColorJitterPro(
        #     brightness=[0.8, 1.2] if soft else [0.6, 1.4],
        #     contrast=[0.85, 1.15] if soft else [0.7, 1.3],
        #     saturation=[0.75, 1.25] if soft else [0.5, 1.5],
        #     hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
        #     gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        # ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=0
        )
    ]
    if cotta_augs:
        tta_transforms += [transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
                           transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           GaussianNoise(0, gaussian_std)]
    else:
        tta_transforms += [transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5)]

    return transforms.Compose(tta_transforms)

class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha

    def __call__(self, x, x_ema):
        return -(1-self.alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - self.alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

class SoftLikelihoodRatio(torch.nn.Module):
    def __init__(self, clip=0.99, eps=1e-5):
        super(SoftLikelihoodRatio, self).__init__()
        self.eps = eps
        self.clip = clip

    def __call__(self, logits):
        probs = logits.softmax(1)
        probs = torch.clamp(probs, min=0.0, max=self.clip)
        return - (probs * torch.log((probs / (torch.ones_like(probs) - probs)) + self.eps)).sum(1)

@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x

@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update

class ROID(ERM):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, tent_optimizer, n_steps,
                 distributed=False, pretrained=False, episodic=False, debug=False, aug=False, early_stopping=True, num_classes=8,
                 use_weighting=True, prior_correction=True, use_consistency=True, temperature=1/3, momentum_probs=0.9, momentum_src=0.99, img_size=256,
                 batch_size=64):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, distributed, pretrained, debug, aug, early_stopping)
        self.tent_optimizer_cfg = tent_optimizer
        self.episodic = episodic 
        self.num_classes = num_classes
        self.n_steps = n_steps
        self.use_weighting = use_weighting
        self.use_prior_correction = prior_correction
        self.use_consistency = use_consistency
        self.momentum_src = momentum_src
        self.momentum_probs = momentum_probs
        self.temperature = temperature
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)
        self.tta_transform = get_tta_transforms(self.img_size, padding_mode="reflect", cotta_augs=False)

        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()

    def initialize(self, run_name, steps_per_epoch, weights=None):
        super().initialize(run_name, steps_per_epoch, weights)
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()
        tent_params, names = self.collect_params()
        self.tent_optimizer = instantiate(self.tent_optimizer_cfg, params=tent_params)

    def loss_calculation(self, x):
        outputs = self.model(x)

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - torch.nn.functional.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.softmax_entropy(outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(x[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

        return outputs, loss


    def eval(self, dataloader, stage, state_dict=None):
        if state_dict is not None:
            self.best_model_params = state_dict
            self.load_state_dict(state_dict)

        cumulative_loss, num_elements= 0.0, 0.0
        all_logits, all_y = [], []

        if stage == "test":
            self.configure_model()

            for d in dataloader.values():
                # Reset parameters for every new domain
                self.reset()

                for batch in d:
                    X, y, metadata = batch

                    X = X.to(self.device)

                    if self.episodic:
                        self.reset()
                    
                    for i in range(self.n_steps):
                        logits = self.forward_and_adapt(X)

                    all_logits.append(logits.detach().cpu())
                    all_y.append(y.detach().cpu())

            all_logits, all_y = torch.cat(all_logits), torch.cat(all_y)

            metrics = self.get_metrics(all_logits, all_y)

            
            if self.wandb:
                self.log(metrics, stage)
                if self.debug:
                    self.log_samples(X, y, metadata, stage)

        else:
            # Standard eval
            self.model.eval()

            with torch.no_grad():
                for batch in dataloader:
                    X, y, metadata = batch

                    X = X.to(self.device)
                    y = y.to(self.device)
                    metadata = metadata.to(self.device)

                    logits, loss = self.get_loss(X, y, metadata, stage)

                    batch_size = len(X)
                    num_elements += batch_size

                    cumulative_loss += loss * batch_size

                    all_logits.append(logits)
                    all_y.append(y)
            
            all_logits, all_y = torch.cat(all_logits), torch.cat(all_y)

            loss = cumulative_loss / num_elements
            print(f"Eval loss: {loss}")
            
            metrics = self.get_metrics(all_logits, all_y)
            log_data = {"loss": loss.item()}
            metrics.update(log_data)
            
            if self.wandb:
                self.log(metrics, stage)
                if self.debug:
                    self.log_samples(X, y, metadata, stage)

        return metrics


    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)


    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward

        outputs, loss = self.loss_calculation(x)
        loss.backward()
        self.tent_optimizer.step()
        self.tent_optimizer.zero_grad()

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=self.momentum_src,
            device=self.device
        )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        return outputs


    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
    

    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        for m in self.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None


    def reset(self):
        self.load_model_and_optimizer()


    def load_model_and_optimizer(self):
        """Restore the model and optimizer states from copies."""
        self.model.load_state_dict(self.best_model_params, strict=False)
        tent_params, names = self.collect_params()
        self.tent_optimizer = instantiate(self.tent_optimizer_cfg, params=tent_params)
        