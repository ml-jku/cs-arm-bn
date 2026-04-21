import torch
from hydra.utils import instantiate

from plt.trainer.ERM import ERM
from copy import deepcopy
from plt.utils import EarlyStopping


class DANN(ERM):

    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, domain_loss, featurizer_lr, classifier_lr, domain_classifier_lr, penalty_weight, patience, classifier_epochs=None):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience)
        self.featurizer_lr = featurizer_lr
        self.classifier_lr = classifier_lr
        self.domain_classifier_lr = domain_classifier_lr

        self.domain_loss_cfg = domain_loss

        self.penalty_weight = penalty_weight
  

    def initialize(self, run_name, steps_per_epoch=None, weights=None):
        super().initialize(run_name, steps_per_epoch)
        
        params = [
                    {"params": self.model.featurizer.named_parameters(), "lr": self.featurizer_lr},
                    {"params": self.model.classifier.named_parameters(), "lr": self.classifier_lr},
                    {"params": self.model.domain_classifier.named_parameters(), "lr": self.domain_classifier_lr}
                 ]

        self.optimizer = instantiate(self.optimizer_cfg, params=params, _convert_="object")
        self.loss_fn = instantiate(self.loss_cfg)

        self.domain_loss_fn = instantiate(self.domain_loss_cfg)

        self.total_steps = steps_per_epoch * self.n_epochs

        if self.total_steps > 0 and self.scheduler_cfg:
            self.scheduler = instantiate(self.scheduler_cfg, optimizer=self.optimizer, T_max=self.total_steps)
        else:
            self.scheduler = None   

            
    def get_loss(self, X, y, metadata, stage=False, model=None):
        logits, metadata_logits = self.model(X)

        class_loss = self.loss_fn(logits[:len(y)], y)

        domain_loss = self.domain_loss_fn(metadata_logits, metadata)

        loss = class_loss + domain_loss * self.penalty_weight

        loss_comps = {
            "class_loss": class_loss.item(),
            "domain_loss": domain_loss.item(),
            "total_loss": loss.item()
        }

        return logits, loss, loss_comps


    def eval(self, dataloader, stage, state_dict=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.model.eval()

        cumulative_loss, cumulative_class_loss, cumulative_domain_loss, num_elements= 0.0, 0.0, 0.0, 0.0
    
        all_logits, all_y = [], []

        with torch.no_grad():
            for batch in dataloader:
                X, y, metadata = batch
                
                X = X.to(self.device)
                y = y.to(self.device)
                metadata = metadata.to(self.device)

                logits, loss, loss_comps = self.get_loss(X, y, metadata)

                batch_size = len(X)

                num_elements += batch_size

                cumulative_loss += loss * batch_size
                cumulative_class_loss += loss_comps["class_loss"] * batch_size
                cumulative_domain_loss += loss_comps["domain_loss"] * batch_size

                all_logits.append(logits)
                all_y.append(y)
            
            all_logits, all_y = torch.cat(all_logits), torch.cat(all_y)


            class_loss = cumulative_class_loss / num_elements
            domain_loss = cumulative_domain_loss / num_elements

            loss = cumulative_loss / num_elements
            
            metrics = self.get_metrics(all_logits, all_y)

            metrics.update({
                "loss": loss.item(),
                "class_loss": class_loss,
                "domain_loss": domain_loss,
            })

        if self.wandb:
            self.log(metrics, stage)

        return metrics


    def train_step(self, batch, unlabeled_batch=None, mode="supervised", stage="train", model="full"):
        self.optimizer.zero_grad()

        X, y, metadata = batch

        if unlabeled_batch:
            X_unlabel, _, metadata_unlabel = unlabeled_batch
            X = torch.cat([X, X_unlabel])
            metadata = torch.cat([metadata, metadata_unlabel])

        X = X.to(self.device)
        y = y.to(self.device)
        metadata = metadata.to(self.device)

        if self.aug:
            X, y = self.transform(X, y)

        _, loss, loss_comps = self.get_loss(X, y, metadata, model=model)
            
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler_cfg:
            self.scheduler.step()

        if self.wandb and (self.step % 10) == 0:
            loss_comps.update({"lr": self.optimizer.param_groups[0]['lr']})

            self.log(loss_comps, stage)



    def train_epoch(self, dataloader, unlabeled_loader, mode="supervised", stage="train", model="full"):
        self.model.train()

        for batch, unlabeled_batch in zip(dataloader, unlabeled_loader):
            self.train_step(batch, unlabeled_batch, mode, stage, model=model)

            self.step += 1
    

    def train(self, train_loader, val_loader, unlabeled_loader, domain_dataloaders=None, split=False, outdir="", model=None):
        self.model.to(self.device)

        self.eval(val_loader, stage="val")

        best_metrics = None
        best_model = None
        best_model_params = None

        for epoch in range(self.n_epochs):
            self.train_epoch(train_loader, unlabeled_loader)
            metrics = self.eval(val_loader, stage="val")

            self.early_stopper(metrics["class_loss"])

            if best_metrics is None or metrics["class_loss"] < best_metrics["class_loss"]:
                best_metrics = metrics
                best_model_params = deepcopy(self.model.state_dict())
                best_model = deepcopy(self.model)
            
            if self.early_stopper.early_stop:
                print("Early stopping triggered.")
                return best_metrics, best_model_params, best_model

            self.epoch += 1
            print(f"Epoch {self.epoch}")
        
        self.eval(val_loader, stage="val")

        return best_metrics, best_model_params, best_model
    
    