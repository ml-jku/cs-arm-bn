# Code from https://github.com/DequanWang/tent/blob/master/tent.py
import torch 
from hydra.utils import instantiate
from copy import deepcopy
from plt.trainer.ERM import ERM
import torchvision
import math

from einops import rearrange



class DEYO(ERM):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, tent_optimizer, n_steps, distributed=False,
                 pretrained=False, episodic=False, debug=False, aug=False, early_stopping=True, num_classes=8,
                 REWEIGHT_ENT=True, REWEIGHT_PLPD=True, PLPD=0.2, MARGIN=0.5, margin_e0=0.4, aug_type="patch", occlusion_size=112, row_start=56, column_start=56, patch_len=4):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, distributed, pretrained, debug, aug, early_stopping)
        self.tent_optimizer_cfg = tent_optimizer
        self.episodic = episodic 
        self.n_steps = n_steps

        self.reweight_ent = REWEIGHT_ENT
        self.reweight_plpd = REWEIGHT_PLPD

        self.plpd_threshold = PLPD
        self.deyo_margin = MARGIN * math.log(num_classes)
        self.margin_e0 = margin_e0 * math.log(num_classes)

        self.aug_type = aug_type
        self.occlusion_size = occlusion_size
        self.row_start = row_start
        self.column_start = column_start
        self.patch_len = patch_len


        self.ent = lambda logits: -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

    def loss_calculation(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x.clone()
        outputs = self.model(imgs_test)

        entropys = self.ent(outputs)
        filter_ids_1 = torch.where((entropys < self.deyo_margin))
        entropys = entropys[filter_ids_1]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        x_prime = imgs_test[filter_ids_1]
        x_prime = x_prime.detach()
        if self.aug_type == 'occ':
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.occlusion_size, self.occlusion_size)
            x_prime[:, :, self.row_start:self.row_start + self.occlusion_size, self.column_start:self.column_start + self.occlusion_size] = occlusion_window
        elif self.aug_type == 'patch':
            resize_t = torchvision.transforms.Resize(((imgs_test.shape[-1] // self.patch_len) * self.patch_len, (imgs_test.shape[-1] // self.patch_len) * self.patch_len))
            resize_o = torchvision.transforms.Resize((imgs_test.shape[-1], imgs_test.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.patch_len, ps2=self.patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0], x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1), perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.patch_len, ps2=self.patch_len)
            x_prime = resize_o(x_prime)
        elif self.aug_type == 'pixel':
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:, :, torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=imgs_test.shape[-1], ps2=imgs_test.shape[-1])

        with torch.no_grad():
            outputs_prime = self.model(x_prime)

        prob_outputs = outputs[filter_ids_1].softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim=1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1, 1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1, 1))
        plpd = plpd.reshape(-1)

        filter_ids_2 = torch.where(plpd > self.plpd_threshold)
        entropys = entropys[filter_ids_2]
        if len(entropys) == 0:
            loss = None  # set loss to None, since all instances have been filtered
            return outputs, loss

        plpd = plpd[filter_ids_2]

        if self.reweight_ent or self.reweight_plpd:
            coeff = (float(self.reweight_ent) * (1. / (torch.exp(((entropys.clone().detach()) - self.margin_e0)))) +
                     float(self.reweight_plpd) * (1. / (torch.exp(-1. * plpd.clone().detach())))
                     )
            entropys = entropys.mul(coeff)

        loss = entropys.mean(0)
        return outputs, loss

    def initialize(self, run_name, steps_per_epoch, weights=None):
        super().initialize(run_name, steps_per_epoch, weights)
        tent_params, names = self.collect_params()
        self.tent_optimizer = instantiate(self.tent_optimizer_cfg, params=tent_params)


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
        outputs, loss = self.loss_calculation(x)
        # update model only if not all instances have been filtered
        if loss is not None:
            loss.backward()
            self.tent_optimizer.step()
        self.tent_optimizer.zero_grad()
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
        