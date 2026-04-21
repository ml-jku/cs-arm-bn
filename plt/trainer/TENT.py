# Code from https://github.com/DequanWang/tent/blob/master/tent.py
import torch 
from hydra.utils import instantiate
from copy import deepcopy
from plt.trainer.ERM import ERM

class TENT(ERM):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, tent_optimizer, n_steps, distributed=False, pretrained=False, episodic=False, debug=False, aug=False, early_stopping=True):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, distributed, pretrained, debug, aug, early_stopping)
        self.tent_optimizer_cfg = tent_optimizer
        self.episodic = episodic 
        self.n_steps = n_steps

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
        # forward
        outputs = self.model(x)

        # adapt
        loss = self.softmax_entropy(outputs).mean(0)
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
        