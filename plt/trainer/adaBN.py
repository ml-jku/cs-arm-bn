import torch 
import numpy as np 

from plt.trainer.ERM import ERM
from plt.utils import update_bn_with_full_domain

class AdaBN(ERM):
    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, first_eval=True, pretrained=False, full_domain_stats=False, distributed=False, debug=False, aug=False, early_stopping=True):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, first_eval, distributed, pretrained, debug, aug, early_stopping)
        self.full_domain_stats = full_domain_stats

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
            self.model.train() if stage == "test" else self.model.eval()

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