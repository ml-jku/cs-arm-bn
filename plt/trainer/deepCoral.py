import torch
from hydra.utils import instantiate

from plt.trainer.ERM import ERM
from plt.utils import split_by_groups
from copy import deepcopy
from plt.utils import EarlyStopping


class DeepCoral(ERM):
    def __init__(self, model, optimizer, loss, n_epochs, scheduler, wandb, patience, penalty_weight):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience)
        self.penalty_weight = penalty_weight
        print(self.scheduler_cfg)


    def get_loss(self, X, y, metadata, stage=False, model=None):
        features, logits = self.model(X)
        class_loss = self.loss_fn(logits[:len(y)], y)
        
        groups = metadata

        penalty = self.penalty_by_groups(features, groups)

        loss = class_loss + penalty * self.penalty_weight

        loss_comps = {
                      "class_loss": class_loss.item(),
                      "penalty" : penalty.item(),
                      "total_loss" : loss.item()
            }

        return logits, loss, loss_comps


    def coral_penalty(self, x, y):

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)

        cent_x = x - mean_x
        cent_y = y - mean_y

        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff



    def eval(self, dataloader, stage, state_dict=None, model=None):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.model.eval()

        cumulative_loss, cumulative_class_loss, cumulative_penalty, num_elements= 0.0, 0.0, 0.0, 0.0
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
                cumulative_penalty += loss_comps["penalty"] * batch_size

                all_logits.append(logits)
                all_y.append(y)
            
            all_logits, all_y = torch.cat(all_logits), torch.cat(all_y)

            class_loss = cumulative_class_loss / num_elements
            penalty = cumulative_penalty / num_elements

            loss = cumulative_loss / num_elements
            print(f"Eval loss: {loss}")
            
            metrics = self.get_metrics(all_logits, all_y)

            metrics.update({"loss": loss.item(),
                            "class_loss" : class_loss,
                            "penalty": penalty})

        if self.wandb:
            self.log(metrics, stage)

        return metrics

    
    def penalty_by_groups(self, features, groups):
        group_indices, unique_groups, _ = split_by_groups(groups)
        
        n_groups = unique_groups.numel()

        penalty = torch.zeros(1, device=self.device)

        if n_groups > 1:
            for i, i_group in enumerate(unique_groups):
                for j in range(i+1, n_groups):
                    j_group = unique_groups[j]
                    penalty += self.coral_penalty(features[group_indices[int(i_group)]], features[group_indices[int(j_group)]])
            
            penalty /= (n_groups * (n_groups-1) / 2)
        
        return penalty


    def train_epoch(self, dataloader, unlabeled_loader, mode="supervised", stage="train", model="full"):
        self.model.train()

        for batch, unlabeled_batch in zip(dataloader, unlabeled_loader):
            self.train_step(batch, unlabeled_batch, mode, stage, model=model)

            self.step += 1


    def train(self, train_loader, val_loader, unlabeled_loader, domain_dataloader, split=None, outdir=None):
        self.model.to(self.device)

        self.eval(val_loader, stage="val")

        best_metrics = None
        best_model = None

        for epoch in range(self.n_epochs):
            self.train_epoch(train_loader, unlabeled_loader)
            metrics = self.eval(val_loader, stage="val")

            self.early_stopper(metrics["class_loss"])

            if best_metrics is None or metrics["class_loss"] < best_metrics["loss"]:
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
