import os
import torch 
from copy import deepcopy
from hydra.utils import instantiate
from plt.trainer.ERM import ERM
from plt.utils import update_bn_with_full_domain


class BEN(ERM):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, full_domain_stats, first_eval, distributed=False, pretrained=False, acc_steps=None, debug=False, aug=False, early_stopping=True, cyber=False, prior_mean_w=4, prior_var_w=4):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, first_eval, distributed, pretrained, debug, aug, early_stopping)
        self.full_domain_stats = full_domain_stats
        self.acc_steps = acc_steps
        self.cyber = cyber 
        self.prior_mean_w = prior_mean_w
        self.prior_var_w = prior_var_w

    def initialize(self, run_name, steps_per_epoch, weights=None):
        self.all_steps_per_epoch = steps_per_epoch
        self.steps_per_epoch = (steps_per_epoch // self.acc_steps) 
        self.total_steps = self.n_epochs * self.steps_per_epoch  

        super().initialize(run_name, self.steps_per_epoch, weights)


    def get_loss(self, X, y, metadata, stage=False):
        X_pos = X[torch.where(~y.isnan())]
        X_neg = X[torch.where(y.isnan())]

        self.model.train()

        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.reset_running_stats()

        with torch.no_grad():
            _ = self.model(X_neg)

        self.model.eval()
        logits = self.model(X_pos)        
        
        # remove unlabeled data 
        mask = ~y.isnan()
        y = y[mask].to(torch.int64)

        loss = self.loss_fn(logits, y)
        
        return logits, loss
    
    

    def train_step(self, batch, all_loss, i, stage="train"):
        X, y, metadata = batch

        X = X.to(self.device)
        y = y.to(self.device)

        _, loss = self.get_loss(X, y, metadata, stage)

        all_loss.append(loss)

        loss = loss / self.acc_steps 
        loss.backward()
        
        if ((i+1) % self.acc_steps) == 0:
            avg_loss = sum(all_loss) / len(all_loss)
            
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.wandb: 
                log_data = {"loss": avg_loss,
                            "lr": self.optimizer.param_groups[0]['lr']}

                self.log(log_data, stage)
            
        return all_loss


    def train_epoch(self, dataloader, stage="train"):
        for i, batch in enumerate(dataloader):

            if (i % self.acc_steps == 0):
                all_loss = []

            all_loss = self.train_step(batch, all_loss, i, stage)

            self.step += 1
        
        # Discard last batch if not intended size
        self.optimizer.zero_grad()


    def eval(self, dataloader, stage, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.prepare()

        cumulative_loss, num_elements= 0.0, 0.0
        all_logits, all_y = [], []

        if self.full_domain_stats and stage == "test":
            for loader in dataloader.values():
                update_bn_with_full_domain(loader, self.model, self.device)

                self.model.eval()

                with torch.no_grad():
                    for batch in loader:
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
        else:
            self.model.train()

            with torch.no_grad():
                for batch in dataloader:

                    X, y, metadata = batch

                    X = X.to(self.device)
                    y = y.to(self.device)

                    logits, loss = self.get_loss(X, y, metadata, stage)

                    batch_size = len(X)
                    num_elements += batch_size

                    cumulative_loss += loss * batch_size

                    mask = ~y.isnan()
                    y = y[mask].to(torch.int64)

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