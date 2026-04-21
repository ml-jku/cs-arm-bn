import torch
from hydra.utils import instantiate
from plt.trainer.ERM import ERM


class ARM_CML(ERM):

    def __init__(self, model, optimizer, loss, scheduler, n_epochs, wandb, patience, context_net_cfg, n_context_channels, adapt_bn, acc_steps, debug=False, aug=False, early_stopping=True, first_eval=False, distributed=False, pretrained=False):
        super().__init__(model, optimizer, loss, scheduler, n_epochs, wandb, patience, first_eval, distributed, pretrained, debug, aug, early_stopping)

        self.context_net_cfg = context_net_cfg
        self.n_context_channels = n_context_channels
        self.adapt_bn = adapt_bn
        self.acc_steps = acc_steps


    def initialize(self, run_name, steps_per_epoch=None, weights=None):
        super().initialize(run_name, steps_per_epoch)
        self.context_net = instantiate(self.context_net_cfg)
        
        params = list(self.model.parameters()) + list(self.context_net.parameters())
        self.optimizer = instantiate(self.optimizer_cfg, params=params, _convert_="object")

        self.total_steps = steps_per_epoch * self.n_epochs

        if self.total_steps > 0 and self.scheduler_cfg:
            self.scheduler = instantiate(self.scheduler_cfg, optimizer=self.optimizer, T_max=self.total_steps)
        else:
            self.scheduler = None   


    def prepare(self):
        super().prepare()
        self.context_net.to(self.device)


    def get_loss(self,  X, y, metadata, stage):   
        batch_size, c, h, w = X.shape

        # IMPORTANT: if sampling more than one domain per batch, uncomment lines 51-54 
        domains, counts = torch.unique(metadata, return_counts=True)
        meta_batch_size = len(domains)
        samples_per_group = torch.unique(counts)

        if len(samples_per_group) > 1:
            print("unequal support sizes")

        support_size = samples_per_group[0].to(self.device)

        if self.adapt_bn:
            # logits = []
            # for i in range(meta_batch_size):
                # X_i = X[i*support_size:(i+1)*support_size]
            context_i = self.context_net(X)
            context_i = context_i.mean(dim=0).expand(support_size, -1, -1, -1)
            X = torch.cat([X, context_i], dim=1)
            logits = self.model(X)

            loss = self.loss_fn(logits, y)
        
        else:
            context = self.context_net(X) # Shape: batch_size, channels, H, W
            context = context.reshape((meta_batch_size, support_size, self.n_context_channels, h, w))
            context = context.mean(dim=1) # Shape: meta_batch_size, self.n_context_channels
            context = torch.repeat_interleave(context, repeats=support_size, dim=0) # meta_batch_size * support_size, context_size
            X = torch.cat([X, context], dim=1)
            logits = self.model(X)

            loss = self.loss_fn(logits, y)
        
        return logits, loss

    def train_step(self, batch, all_loss, i, stage="train"):
        X, y, metadata = batch

        X = X.to(self.device)
        y = y.to(self.device)
        # metadata = metadata.to(self.device)

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
        self.prepare()
        self.model.train()

        for i, batch in enumerate(dataloader):

            if (i % self.acc_steps == 0):
                all_loss = []

            all_loss = self.train_step(batch, all_loss, i, stage)

            self.step += 1
        
        # Discard last batch if not intended size
        self.optimizer.zero_grad()