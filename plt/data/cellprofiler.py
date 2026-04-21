import os
import torch
import math 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit, GroupKFold, StratifiedKFold

from plt.utils import seed_worker
from plt.data.sampler import GroupBatchSampler

from torch.utils.data import Dataset, DataLoader


class CellProfiler(Dataset):
    def __init__(self, profiles, labels, metadata):
        self.profiles = profiles
        self.labels = labels 
        self.metadata = metadata

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        profile = self.profiles[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        return profile, label, metadata


class CellProfilerDataModule():
    def __init__(self, X, y, metadata, workers, metadata_name, batch_size, splitters, seed, new_wells, groups_per_batch, groups_per_batch_eval, standardize=True, batch_size_eval=None):
        self.X = X
        self.y = y
        self.metadata = metadata
        self.new_wells = new_wells 

        self.metadata_name = metadata_name

        self.seed = seed
        self.standardize = standardize
        self.workers = workers
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval else batch_size
        self.groups_per_batch = groups_per_batch
        self.groups_per_batch_eval = groups_per_batch_eval

        self.outer_splitter = self.set_splitter(**splitters["outer"])
        self.inner_splitter = self.set_splitter(**splitters["inner"])

        self.num_splits = self.outer_splitter.n_splits * self.inner_splitter.n_splits


    def setup(self, split):
        self.train_dataloader = self.splits[split]["train"]
        self.val_dataloader = self.splits[split]["val"]
        self.test_dataloader = self.splits[split]["test"]
        self.unlabeled_dataloader = self.splits[split]["unlabeled"] 
        self.domain_dataloaders = self.splits[split]["domain_loaders"]


    def set_splitter(self, n_splits, cv, group, shuffle, ratio_test=None): 
        if cv:
            if group:
                splitter = GroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)
            else:
                splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.seed)
        else:
            if group:
                splitter = GroupShuffleSplit(test_size=ratio_test, n_splits=n_splits, random_state=self.seed)
            else:
                splitter = StratifiedShuffleSplit(test_size=ratio_test, n_splits=n_splits, random_state=self.seed)
        
        splitter.group = group

        return splitter
    

    def fit_encoder(self, X, encoder, variable_name=None):
        if variable_name:
            X = X[variable_name]
        encoder.fit(X)


    def encode(self, X, encoder, variable_name=None):
        if variable_name:
            X = X[variable_name]
        labels = encoder.transform(X)
        return labels


    def preprocess_data(self):
        self.splits = {}

        X = np.load(self.X)
        y = np.load(self.y, allow_pickle=True)
        metadata_index = pd.read_parquet(self.metadata)

        self.label_encoder = LabelEncoder()
        self.metadata_encoder = LabelEncoder()

        self.fit_encoder(y, self.label_encoder)
        self.fit_encoder(metadata_index, self.metadata_encoder, self.metadata_name)

        y = self.encode(y, self.label_encoder)
        metadata = self.encode(metadata_index, self.metadata_encoder, self.metadata_name)


        for n_outer_fold, (trainval_idx, test_idx) in enumerate(self.outer_splitter.split(X, y=y, groups=metadata_index[self.outer_splitter.group])):
            X_trainval, X_test = X[trainval_idx, :], X[test_idx, :]
            y_trainval, y_test = y[trainval_idx], y[test_idx]
            metadata_trainval, metadata_test = metadata[trainval_idx], metadata[test_idx]

            metadata_index_trainval = metadata_index.iloc[trainval_idx]
            metadata_index_test = metadata_index.iloc[test_idx]

            metadata_index_trainval.reset_index(inplace=True)
            metadata_index_test.reset_index(inplace=True)
            
            if self.new_wells:
                X_trainval, y_trainval, metadata_trainval, metadata_index_trainval, X_test, y_test, metadata_test, metadata_index_test = self.filter_new_wells(metadata_index_trainval, X_trainval, y_trainval, metadata_trainval, metadata_index_test, X_test, y_test, metadata_test)

            for n_inner_fold, (train_idx, val_idx) in enumerate(self.inner_splitter.split(X_trainval, y=y_trainval, groups=metadata_index_trainval[self.inner_splitter.group])):
                X_train, X_val = X_trainval[train_idx, :], X_trainval[val_idx, :]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
                metadata_train, metadata_val = metadata_trainval[train_idx], metadata_trainval[val_idx]

                metadata_index_train = metadata_index_trainval.iloc[train_idx]
                metadata_index_val = metadata_index_trainval.iloc[val_idx]

                metadata_index_train.reset_index(inplace=True)
                metadata_index_val.reset_index(inplace=True)
                
                if self.new_wells:
                    X_train, y_train, metadata_train, metadata_index_train, X_val, y_val, metadata_val, metadata_index_val = self.filter_new_wells(metadata_index_train, X_train, y_train, metadata_train, metadata_index_val, X_val, y_val, metadata_val)

                if self.standardize: 
                    scaler = StandardScaler()

                    X_train = scaler.fit_transform(X_train)
                    X_val = scaler.transform(X_val)
                    X_test = scaler.transform(X_test)

                train_batch_sampler = GroupBatchSampler(metadata_train, self.batch_size, self.groups_per_batch, eval=False) if self.groups_per_batch else None
                train_loader = self.get_dataloader(X_train, y_train, metadata_train, is_train=True, batch_sampler=train_batch_sampler)
                val_batch_sampler = GroupBatchSampler(metadata_val, self.batch_size_eval, self.groups_per_batch_eval, eval=True) if self.groups_per_batch_eval else None
                val_loader = self.get_dataloader(X_val, y_val, metadata_val, is_train=False, batch_sampler=val_batch_sampler)
                test_batch_sampler = GroupBatchSampler(metadata_test, self.batch_size_eval, self.groups_per_batch_eval, eval=True) if self.groups_per_batch_eval else None
                test_loader = self.get_dataloader(X_test, y_test, metadata_test, is_train=False, batch_sampler=test_batch_sampler)


                # TODO: add support for inner cross validation
                self.splits[n_outer_fold] = {
                    "train": train_loader,
                    "val": val_loader,
                    "test": test_loader,
                    "unlabeled": None,
                    "domain_loaders" : None
                    }
                    


    def get_dataloader(self, X, y, metadata, is_train, batch_sampler):
        dataset = CellProfiler(X, y, metadata)
        shuffle = is_train

        batch_size = self.batch_size if is_train else self.batch_size_eval

        g = torch.Generator()
        g.manual_seed(self.seed)

        if not batch_sampler:
            dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=is_train,
                    worker_init_fn=seed_worker,
                    generator=g,
                    prefetch_factor=8,
                    persistent_workers=True
                )
        else:
            dataloader = DataLoader(
                        dataset,
                        num_workers=self.workers,
                        pin_memory=True,
                        worker_init_fn=seed_worker,
                        generator=g,
                        prefetch_factor=8,
                        persistent_workers=True,
                        batch_sampler=batch_sampler
            )
        
        return dataloader
    

    def filter_new_wells(self, metadata_index_train, X_train, y_train, metadata_train, metadata_index_test,  X_test, y_test, metadata_test):
        test_well_per_mol = []
        groups = []

        for g, df in metadata_index_train.groupby("Metadata_JCP2022"):
            print(g)
            groups.append(g)    
            test_well_per_mol.append(df["Metadata_Well"].unique().tolist()[-1])

        metadata_index_test = metadata_index_test[metadata_index_test["Metadata_Well"].isin(test_well_per_mol)]
        metadata_index_train = metadata_index_train[~metadata_index_train["Metadata_Well"].isin(test_well_per_mol)]

        X_train, X_test = X_train[metadata_index_train.index], X_test[metadata_index_test.index]
        y_train, y_test = y_train[metadata_index_train.index], y_test[metadata_index_test.index]
        metadata_train, metadata_test = metadata_train[metadata_index_train.index], metadata_test[metadata_index_test.index]
        
        metadata_index_train.reset_index(inplace=True, drop=True), metadata_index_test.reset_index(inplace=True, drop=True)
        
        return X_train, y_train, metadata_train, metadata_index_train,  X_test, y_test, metadata_test, metadata_index_test