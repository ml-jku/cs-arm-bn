import os 
import torch 
import numpy as np
import wandb
import datetime
from plt.utils import _make_unique_outdir

class CrossValidator():   
    def __init__(self, datamodule, trainer, pretrained_models=None):
        self.datamodule = datamodule
        self.datamodule.preprocess_data()
        self.pretrained_models = pretrained_models

        self.trainer = trainer

        run_name = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}_{self.trainer.__class__.__name__}'
        self.outdir = _make_unique_outdir("results", run_name)


    def cross_validate(self):
        agg_results = {}
        results = {}

        for split in range(self.datamodule.num_splits):
            self.datamodule.setup(split=split)

            train_loader = self.datamodule.train_dataloader
            val_loader = self.datamodule.val_dataloader
            test_loader = self.datamodule.test_dataloader
            unlabeled_loader = self.datamodule.unlabeled_dataloader
            domain_dataloaders = self.datamodule.domain_dataloaders

            pretrained_weights = self.pretrained_models[split] if self.pretrained_models else None
            self.trainer.initialize(f"Outer_{split}/Inner_0", len(train_loader), weights=pretrained_weights)

            if self.trainer.n_epochs != 0:
                _, best_model_params, _ = self.trainer.train(train_loader, val_loader, unlabeled_loader, domain_dataloaders, split=split, outdir=self.outdir)
            else:
                best_model_params = torch.load(self.pretrained_models[split]) if self.pretrained_models else None

            test_metrics = self.trainer.eval(test_loader, "test", best_model_params)

            results[f"Outer_{split}/Inner_0/test"] = test_metrics
            
        agg_results["Test_Acc_Mean"] = np.mean([results[f"Outer_{i}/Inner_0/test"]["accuracy"] for i in range(self.datamodule.num_splits)])
        agg_results["Test_Acc_Std"] = np.std([results[f"Outer_{i}/Inner_0/test"]["accuracy"] for i in range(self.datamodule.num_splits)])

        # TODO: set wandb to optional
        wandb.log(agg_results)

        return agg_results 
        
