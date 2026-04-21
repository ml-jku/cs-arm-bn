import os
import wandb
import torch

from copy import deepcopy
from hydra.utils import instantiate
from sklearn.metrics import accuracy_score

from plt.trainer.base import Trainer
from plt.utils import EarlyStopping

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from torchvision.transforms import v2
from torchvision.utils import make_grid

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class ERM(Trainer):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, first_eval=True, distributed=False, pretrained=False, debug=False, aug=False, early_stopping=True, classifier_epochs=0):
        self.model_cfg = model
        self.optimizer_cfg = optimizer
        self.loss_cfg = loss
        self.scheduler_cfg = scheduler

        self.n_epochs = n_epochs
        self.wandb = wandb
        self.distributed = distributed

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.aug = aug

        if self.aug == "cutmix":
            # TODO fix
            self.model_cfg.output_dim = 10
            self.transform = v2.CutMix(num_classes=self.model_cfg.output_dim) 
        elif self.aug == "mixup":
            self.mixup = v2.MixUp(num_classes=self.model_cfg.output_dim) 
        
        self.early_stopping = early_stopping
        self.patience = patience    
        self.classifier_epochs = classifier_epochs
        self.debug = debug
        self.pretrained = pretrained
        self.first_eval = first_eval


    def initialize(self, run_name, steps_per_epoch, weights=None):
        self.steps_per_epoch = steps_per_epoch
        self.model = instantiate(self.model_cfg)

        if self.distributed:
            rank = dist.get_rank()
            
            self.device = rank % torch.accelerator.device_count()
            self.model = self.model.to(self.device)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.device])
        else:
            self.model = self.model.to(self.device)
        
        if weights:
            weights = torch.load(weights)
            self.model.load_state_dict(weights, strict=False) 

        self.optimizer = instantiate(self.optimizer_cfg, params=self.model.parameters())
        self.loss_fn = instantiate(self.loss_cfg)

        if self.debug and self.wandb:
            wandb.watch(self.model, self.loss_fn, log="all", log_freq=100)

        self.total_steps = self.steps_per_epoch * self.n_epochs

        if self.total_steps > 0 and self.scheduler_cfg:
            if self.scheduler_cfg["_target_"].startswith("torch"):
                self.scheduler = instantiate(self.scheduler_cfg, optimizer=self.optimizer, T_max=self.total_steps)
            elif self.scheduler_cfg["_target_"].startswith("transformers"):
                warmup_epochs = self.scheduler_cfg["warmup_epochs"]
                del self.scheduler_cfg["warmup_epochs"]
                self.scheduler = instantiate(self.scheduler_cfg, optimizer=self.optimizer, num_warmup_steps=warmup_epochs*self.steps_per_epoch, num_training_steps=self.total_steps)
        else:
            self.scheduler = None   

        self.epoch = 0
        self.step = 0

        self.run_name = run_name

        if self.early_stopping:
            self.early_stopper = EarlyStopping(patience=self.patience)
    

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict, strict=False)


    def train_step(self, batch, stage="train"):
        self.optimizer.zero_grad()

        X, y, metadata = batch

        X = X.to(self.device)
        y = y.to(self.device)
        metadata = metadata.to(self.device)

        if self.aug:
            X, y = self.transform(X, y)

        _, loss = self.get_loss(X, y, metadata, stage)
            
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if self.wandb and (self.step % 10) == 0: #and int(os.environ["LOCAL_RANK"] == 0):
            log_data = {"loss": loss.item(),
                        "lr": self.optimizer.param_groups[0]['lr']}

            self.log(log_data, stage)


        if self.wandb and (self.step % 10) == 0 and self.debug:
            self.log_samples(X, y, metadata, stage)


    def train_epoch(self, dataloader, stage="train"):
        self.model.train()

        for batch in dataloader:
            self.train_step(batch, stage)

            self.step += 1

    def prepare(self):
        self.model.to(self.device)


    def train(self, train_loader, val_loader, inner_dataloaders=False, stri="", split=False, outdir=""):           
        if val_loader and self.first_eval:
            metrics = self.eval(val_loader, stage="val")

        best_metrics, best_model, best_model_params = None, None, None

        for epoch in range(self.n_epochs):
            self.train_epoch(train_loader)

            if val_loader:
                metrics = self.eval(val_loader, stage="val")

                if self.early_stopping:
                    self.early_stopper(metrics["loss"])

                    if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
                        best_metrics = metrics
                        best_model_params = deepcopy(self.model.state_dict())
                        best_model = deepcopy(self.model)
                        self.best_model_params = best_model_params


                    if self.early_stopper.early_stop:
                        print("Early stopping triggered.")
                        break
                else:
                    best_metrics = None
                    best_model_params = deepcopy(self.model.state_dict())
                    best_model = deepcopy(self.model)

                    self.best_model_params = best_model_params

            self.epoch += 1
            print(f"Epoch {self.epoch}")
        
        fout = os.path.join(outdir, f"{self.__class__.__name__}_epoch{self.epoch}_split{split}.pt")
        torch.save(best_model_params, fout)
        
        return best_metrics, best_model_params, best_model


    def eval(self, dataloader, stage, state_dict=None, inner_dataloaders=None, save=False):
        if state_dict is not None:
            self.load_state_dict(state_dict)

        self.prepare()
        self.model.eval()

        cumulative_loss, num_elements= 0.0, 0.0
        all_logits, all_y = [], []

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

                mask = ~y.isnan()
                y = y[mask].to(torch.int64)
                
                all_logits.append(logits)
                all_y.append(y)
                
                if self.debug and self.wandb:
                    self.log_samples(X, y, metadata, stage)

            
            all_logits, all_y = torch.cat(all_logits), torch.cat(all_y)

            loss = cumulative_loss / num_elements
            print(f"Eval loss: {loss}")
            
            metrics = self.get_metrics(all_logits, all_y)
            log_data = {"loss": loss.item()}
            metrics.update(log_data)

        if self.wandb:
            self.log(metrics, stage)

        return metrics

            
    def get_metrics(self, logits, labels):
        metrics = {}

        _, preds = logits.max(dim=1)

        metrics["accuracy"] = accuracy_score(labels.cpu(), preds.cpu())

        return metrics


    def log(self, log_data, stage):
        step = self.step if stage == "train" else self.epoch
        
        for name, val in log_data.items():
            full_name = f"{self.run_name}/{stage}/{name}"
            wandb.log({full_name: val, 'step': self.step})


    def get_loss(self, X, y, metadata, stage=False):
        logits = self.model(X)

        mask = ~y.isnan()
        y = y[mask].to(torch.int64)

        mask_expanded = mask.unsqueeze(1).expand_as(logits)
        logits = logits[mask_expanded].view(-1, logits.shape[-1])
    
        loss = self.loss_fn(logits, y)

        return logits, loss
    

    def log_samples(self, X, y, metadata, stage):
        imgs = []

        for Xi in X[:min(16, len(X))]:

            im = Xi.cpu() * 255
            imgs.append(im)

        imgs = torch.stack(imgs)  # shape: (N, C, H, W)

        grid = make_grid(imgs, normalize=True)

        wandb.log({f"{stage}/sample": [wandb.Image(grid)]})


