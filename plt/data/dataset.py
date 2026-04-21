import os
import torch
import numpy as np
import pandas as pd
from typing import Union
from tqdm import tqdm 

import tifffile
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, GroupKFold, StratifiedKFold

from plt.data.transforms import TVN, _transform
from plt.utils import calc_mean_std, calc_mean_std_per_domain, seed_worker
from plt.data.sampler import GroupBatchSampler, LabelShiftGroupBatchSampler

from huggingface_mae import MAEModel


class CellPainting(Dataset):
    def __init__(self, img_dir: Union[str, list] = None, 
                        sample_df: pd.DataFrame = None, 
                        labels : list = None, 
                        metadata= None, 
                        transforms=None, 
                        channel_subset=None,
                        *args,
                        **kwargs):

        """ Read samples from cellpainting dataset."""
        sample_list = sample_df["Metadata_Sample_ID"].tolist()

        if isinstance(img_dir, str):
            assert (os.path.isdir(img_dir))
            dirlist = [img_dir] * len(sample_list)

        if isinstance(img_dir, list):
            dirlist = img_dir 

        format_ = "tiff" 
        filelist = [os.path.join(d, f"{key}.{format_}") for d, key in zip(dirlist, sample_list)]

        self.transforms = transforms
        self.labels = labels
        self.channel_subset = channel_subset
        self.filelist = filelist
        self.metadata = metadata


    def __len__(self):
        return len(self.filelist)


    def __getitem__(self, idx):
        img = self.read_img(idx)
        label = self.labels[idx] if self.labels is not None else torch.nan
        metadata = self.metadata[idx]

        if self.transforms and len(self.transforms) > 1:
             img = self.transforms[1](img, metadata)

        return img, label, metadata


    def read_img(self, idx):
        key = self.filelist[idx]
        X = self.load_view(key)

        if self.transforms:
            X = self.transforms[0](X)

        return X


    def load_view(self, file):
        """Load all channels for one sample"""
        image = torch.from_numpy(tifffile.imread(file, maxworkers=5))
        image = image.permute(2, 0, 1)

        if self.channel_subset:
            image = image[self.channel_subset]
        
        return image


class EmbeddingDataset(Dataset):
    def __init__(self, features, labels, metadata):
        self.features = features
        self.labels = labels
        self.metadata = metadata 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = self.features[idx]
        labels = self.labels[idx] if self.labels is not None else torch.nan
        metadata = self.metadata[idx]

        return features, labels, metadata


   
class CellPaintingDataModule():
    def __init__(self, index_file, variable_name, metadata_name, batch_size, new_wells, channels, train_res, test_res, splitters, seed, preprocess, normalize, source, batches, embeddings, rotation, unlabeled, groups_per_batch, groups_per_batch_eval, workers, label_shift=False, shift=None, alpha=None, imb_ratio=False, total_size=None,groups=None, distributed=False, domains_to_filter=None, phase=None, dataloader_per_domain=False, eval_dataloader_per_domain=False, test_dataloader_per_domain=False, negative_index=None, batch_size_eval=None, batch_size_test=None, filter_metadata=False, batch_size_per_domain=False, loader_with_controls=False, neg_batch_size=False, test_index=None):
        self.seed = seed
        self.workers = workers
        self.index_file = index_file    
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval if batch_size_eval else batch_size
        self.batch_size_test = batch_size_test if batch_size_test else self.batch_size_eval
        self.neg_batch_size = neg_batch_size
        self.variable_name = variable_name
        self.metadata_name = metadata_name
        self.new_wells = new_wells
        self.channels = channels
        self.train_res = train_res
        self.test_res = test_res
        self.preprocess = preprocess
        self.normalize = normalize
        self.source = source
        self.batches = batches 
        self.embeddings = embeddings
        self.rotation = rotation 
        self.unlabeled = unlabeled
        self.groups_per_batch = groups_per_batch
        self.groups_per_batch_eval = groups_per_batch_eval
        self.dataloader_per_domain = dataloader_per_domain
        self.eval_dataloader_per_domain = eval_dataloader_per_domain
        self.test_dataloader_per_domain = test_dataloader_per_domain
        self.leave_out_metadata = filter_metadata
        self.domains_to_filter = domains_to_filter
        self.phase = phase
        self.batch_size_per_domain = batch_size_per_domain
        self.distributed = distributed 
        self.loader_with_controls = loader_with_controls
        self.test_index = test_index

        self.label_shift = label_shift
        self.imb_ratio = imb_ratio
        self.total_size = total_size
        self.shift = shift
        self.alpha = alpha

        self.outer_splitter = self.set_splitter(**splitters["outer"])
        self.inner_splitter = self.set_splitter(**splitters["inner"])

        self.negative_index = negative_index

        self.num_splits = self.outer_splitter.n_splits * self.inner_splitter.n_splits

        if groups:
            self.train_group, self.val_group, self.test_group = groups["train"], groups["val"], groups["test"]


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


    def filter_new_wells(self, X_train, y_train, metadata_train, X_test, y_test, metadata_test):
        test_well_per_mol = []
        groups = []

        for g, df in X_train.groupby("Metadata_JCP2022"):
            groups.append(g)    
            test_well_per_mol.append(df["Metadata_Well"].unique().tolist()[-1])
        
        X_test = X_test[X_test["Metadata_Well"].isin(test_well_per_mol)]
        X_train = X_train[~X_train["Metadata_Well"].isin(test_well_per_mol)]

        y_train, y_test = y_train[X_train.index], y_test[X_test.index]

        metadata_train, metadata_test = metadata_train[X_train.index], metadata_test[X_test.index]
        X_train.reset_index(inplace=True, drop=True), X_test.reset_index(inplace=True, drop=True)
        
        return X_train, y_train, metadata_train, X_test, y_test, metadata_test


    def fit_encoder(self, X, encoder, variable_name):
        raw_labels = X[variable_name]
        encoder.fit(raw_labels)


    def encode(self, X, encoder, variable_name):
        raw_labels = X[variable_name]
        labels = encoder.transform(raw_labels)
        return labels
    

    def embed(self, dataloader):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = MAEModel.from_pretrained(self.embeddings.model_path)
        model = model.to(device)

        all_img_features, all_labels, all_metadata = [], [], []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                img, label, metadata = batch

                img_features = model.predict(img.to(device))
                all_img_features.append(img_features)
                all_labels.extend(label)
                all_metadata.extend(metadata)

        all_img_features = torch.cat(all_img_features)

        all_labels = np.array(all_labels)
        all_metadata = np.array(all_metadata)

        return all_img_features.cpu().numpy(), all_labels, all_metadata


    def transform_embeddings(self, train_embeddings, train_metadata, 
                                val_embeddings, val_metadata, 
                                test_embeddings, test_metadata, 
                                negative_embeddings, negative_metadata):
        
        negative_index_train = np.isin(negative_metadata, train_metadata)
        negative_index_val = np.isin(negative_metadata, val_metadata)
        negative_index_test = np.isin(negative_metadata, test_metadata)

        negative_metadata_train = negative_metadata[negative_index_train]
        negative_metadata_val = negative_metadata[negative_index_val]
        negative_metadata_test = negative_metadata[negative_index_test]

        negative_embeddings_train = negative_embeddings[negative_index_train]
        negative_embeddings_val = negative_embeddings[negative_index_val]   
        negative_embeddings_test = negative_embeddings[negative_index_test]

        self.normalizer = TVN()

        if self.embeddings.normalize_embeddings == "TVN_all":
            self.normalizer.fit(negative_embeddings, negative_metadata)
        elif self.embeddings.normalize_embeddings == "TVN_train":
            self.normalizer.fit(negative_embeddings_train, negative_metadata_train)

        self.standardizer = StandardScaler()

        train_embeddings = self.normalizer.transform(train_embeddings, train_metadata, negative_embeddings_train, negative_metadata_train)

        self.standardizer.fit(train_embeddings)
        train_embeddings = self.standardizer.transform(train_embeddings)

        val_embeddings = self.normalizer.transform(val_embeddings, val_metadata, negative_embeddings_val, negative_metadata_val)
        val_embeddings = self.standardizer.transform(val_embeddings)

        test_embeddings = self.normalizer.transform(test_embeddings, test_metadata, negative_embeddings_test, negative_metadata_test)
        test_embeddings = self.standardizer.transform(test_embeddings)

        return train_embeddings, val_embeddings, test_embeddings


    def setup(self, split):
        self.train_dataloader = self.splits[split]["train"]
        self.val_dataloader = self.splits[split]["val"]
        self.test_dataloader = self.splits[split]["test"]
        self.unlabeled_dataloader = self.splits[split]["unlabeled"] 
        self.domain_dataloaders = self.splits[split]["domain_loaders"]


    def preprocess_data(self):
        X = self.read_index(self.index_file)

        self.label_encoder = LabelEncoder()
        self.metadata_encoder = LabelEncoder()

        self.fit_encoder(X, self.label_encoder, self.variable_name)
        self.fit_encoder(X, self.metadata_encoder, self.metadata_name)

        y = self.encode(X, self.label_encoder, self.variable_name)
        metadata = self.encode(X, self.metadata_encoder, self.metadata_name)

        # read negative controls
        if self.negative_index:
            X_neg = self.read_index(self.negative_index)

        self.splits = {}

        for n_outer_fold, (trainval_idx, test_idx) in enumerate(self.outer_splitter.split(X, y=y, groups=X[self.outer_splitter.group] if self.outer_splitter.group in X.columns else None)):
            X_trainval, X_test = X.iloc[trainval_idx], X.iloc[test_idx]
            y_trainval, y_test = y[trainval_idx], y[test_idx]
            metadata_trainval, metadata_test = metadata[trainval_idx], metadata[test_idx]

            if self.test_index:
                X_test = pd.read_parquet(self.test_index)
                y_test = self.encode(X_test, self.label_encoder, self.variable_name)

                known = set(self.metadata_encoder.classes_)
                unknown = [lbl for lbl in X_test[self.metadata_name] if lbl not in known]

                # Update classes_ with new labels (in case our test set has new unseen metadata classes)
                if unknown:
                    self.metadata_encoder.classes_ = np.concatenate([self.metadata_encoder.classes_, unknown])
                    self.metadata_encoder.classes_ = np.sort(self.metadata_encoder.classes_)  # optional: keep classes sorted
                
                metadata_test = self.encode(X_test, self.metadata_encoder, self.metadata_name)

            X_trainval.reset_index(inplace=True, drop=True), X_test.reset_index(inplace=True, drop=True)
            
            if self.negative_index:
                X_neg = X_neg[X_neg[self.metadata_name].isin(self.metadata_encoder.classes_)]
                metadata_neg = self.metadata_encoder.transform(X_neg[self.metadata_name])
                y_neg = np.empty(len(metadata_neg))
                y_neg.fill(np.nan)
            else: 
                metadata_neg = None

            if self.new_wells:
                X_trainval, y_trainval, metadata_trainval, X_test, y_test, metadata_test = self.filter_new_wells(X_trainval, y_trainval, metadata_trainval, X_test, y_test, metadata_test)   

            for n_inner_fold, (train_idx, val_idx) in enumerate(self.inner_splitter.split(X_trainval, y=y_trainval, groups=X_trainval[self.inner_splitter.group] if self.inner_splitter.group in X_trainval.columns else None)):              
                X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
                y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
                metadata_train, metadata_val = metadata_trainval[train_idx], metadata_trainval[val_idx]

                X_train.reset_index(inplace=True, drop=True), X_val.reset_index(inplace=True, drop=True)

                if self.new_wells:
                    X_train, y_train, metadata_train, X_val, y_val, metadata_val = self.filter_new_wells(X_train, y_train, metadata_train, X_val, y_val, metadata_val)   

                if self.normalize == "dataset":
                    preprocess_stats = _transform(self.train_res, self.test_res, is_train=True, normalize=None, preprocess=self.preprocess)
                    preprocess_loader = self.get_cellpainting_dataset(X_train, y_train, metadata_train, preprocess_stats, X_train["img_path"].tolist(), is_train=False)

                    train_stats = calc_mean_std(preprocess_loader)
                    
                elif self.normalize == "batch":
                    preprocess_loaders = self.get_per_domain_dataloaders(X_train, y_train, metadata_train)
                    train_stats = calc_mean_std_per_domain(preprocess_loaders)

                    preprocess_loaders = self.get_per_domain_dataloaders(X_val, y_val, metadata_val)
                    val_stats = calc_mean_std_per_domain(preprocess_loaders)

                    preprocess_loaders = self.get_per_domain_dataloaders(X_test, y_test, metadata_test)
                    test_stats = calc_mean_std_per_domain(preprocess_loaders)

                else:
                    train_stats = None 

                preprocess_train = _transform(self.train_res, self.test_res, is_train=True, normalize=self.normalize, preprocess=self.preprocess, stats=train_stats)

                if self.normalize != "batch":
                    val_stats = train_stats
                    test_stats = train_stats

                preprocess_val = _transform(self.train_res, self.test_res, is_train=False, normalize=self.normalize, preprocess=self.preprocess, stats=val_stats)
                preprocess_test = _transform(self.train_res, self.test_res, is_train=False, normalize=self.normalize, preprocess=self.preprocess, stats=test_stats)

                if self.groups_per_batch:
                    train_batch_sampler = GroupBatchSampler(metadata_train, self.batch_size, self.groups_per_batch, eval=False, negative_groups=metadata_neg, neg_batch_size=self.neg_batch_size)
                else:
                    train_batch_sampler = None 
                
                if self.loader_with_controls:
                    X_train = pd.concat([X_train, X_neg])
                    y_train = np.concatenate([y_train, y_neg])
                    metadata_train = np.concatenate([metadata_train, metadata_neg])
                
                train_loader = self.get_cellpainting_dataset(X_train, y_train, metadata_train, preprocess_train, X_train["img_path"].tolist(), is_train=True, batch_sampler=train_batch_sampler)

                if self.dataloader_per_domain:
                    domain_dataloaders = self.get_per_domain_dataloaders(X_train, y_train, metadata_train, train_stats=train_stats, is_train=True, batch_size=self.batch_size_per_domain)
                else:
                    domain_dataloaders = None

                if self.eval_dataloader_per_domain:
                    val_loader = self.get_per_domain_dataloaders(X_val, y_val, metadata_val, train_stats=train_stats, is_train=False, batch_size=self.batch_size_per_domain)
                else:
                    if self.groups_per_batch_eval:
                        val_batch_sampler = GroupBatchSampler(metadata_val, self.batch_size_eval, self.groups_per_batch_eval, eval=True, negative_groups=metadata_neg, neg_batch_size=self.neg_batch_size)
                    else:
                        val_batch_sampler = None

                    if self.loader_with_controls:
                        X_val = pd.concat([X_val, X_neg])
                        y_val = np.concatenate([y_val, y_neg])
                        metadata_val = np.concatenate([metadata_val, metadata_neg])

                    val_loader = self.get_cellpainting_dataset(X_val, y_val, metadata_val, preprocess_val, X_val["img_path"].tolist(), is_train=False, batch_sampler=val_batch_sampler) 

                if self.test_dataloader_per_domain:
                    test_loader = self.get_per_domain_dataloaders(X_test, y_test, metadata_test, train_stats=train_stats, is_train=False, batch_size=self.batch_size_per_domain, metadata_neg=metadata_neg)
                else:
                    if self.groups_per_batch_eval and self.label_shift:
                        test_batch_sampler = LabelShiftGroupBatchSampler(metadata_test, y_test, self.total_size, self.batch_size_test, self.groups_per_batch_eval, eval=True, shift=self.shift, alpha=self.alpha, imb_ratio=self.imb_ratio, negative_groups=metadata_neg, neg_batch_size=self.neg_batch_size)
                    elif self.groups_per_batch_eval:
                        test_batch_sampler = GroupBatchSampler(metadata_test, self.batch_size_test, self.groups_per_batch_eval, eval=True, negative_groups=metadata_neg, neg_batch_size=self.neg_batch_size) 
                    else:
                        test_batch_sampler=None

                    if self.loader_with_controls:
                        X_test = pd.concat([X_test, X_neg])
                        y_test = np.concatenate([y_test, y_neg])
                        metadata_test = np.concatenate([metadata_test, metadata_neg])
                        
                    test_loader = self.get_cellpainting_dataset(X_test, y_test, metadata_test, preprocess_test, X_test["img_path"].tolist(), is_train=False, batch_sampler=test_batch_sampler)    

                X_testval = pd.concat([X_val, X_test], ignore_index=True)
                metadata_testval = np.concatenate([metadata_val, metadata_test])

                unlabeled_batch_sampler = GroupBatchSampler(metadata_testval, self.batch_size, self.groups_per_batch, eval=False) if self.groups_per_batch else None    
                unlabeled_loader = self.get_cellpainting_dataset(X_testval, None, metadata_testval, preprocess_train, X_testval["img_path"].tolist(), batch_sampler=unlabeled_batch_sampler, is_train=True) if self.unlabeled else None

                if self.embeddings:
                    train_embeddings, y_train, metadata_train = self.embed(train_loader)
                    val_embeddings, y_val, metadata_val = self.embed(val_loader)
                    test_embeddings, y_test, metadata_test = self.embed(test_loader)

                    if self.embeddings.normalize_embeddings:
                        negative_loader = self.get_cellpainting_dataset(X_neg, None, metadata_neg, preprocess_val, X_neg["img_path"].tolist(), is_train=False) if self.negative_index else None
                        negative_embeddings, _, negative_metadata = self.embed(negative_loader) if negative_loader else None
                        train_embeddings, val_embeddings, test_embeddings = self.transform_embeddings(train_embeddings, metadata_train, val_embeddings, metadata_val, test_embeddings, metadata_test, negative_embeddings, negative_metadata)

                    train_loader = self.get_embedding_dataset(train_embeddings, y_train, metadata_train, self.batch_size, is_train=True)
                    val_loader = self.get_embedding_dataset(val_embeddings, y_val, metadata_val, self.batch_size_eval, is_train=False)   
                    test_loader = self.get_embedding_dataset(test_embeddings, y_test, metadata_test, self.batch_size_test, is_train=False)

                self.splits[n_outer_fold] = {
                    "train": train_loader,
                    "val": val_loader,
                    "test": test_loader,
                    "unlabeled": unlabeled_loader,
                    "domain_loaders" : domain_dataloaders
                    }
                
                    
    def read_index(self, index):
        index = pd.read_parquet(index)

        if self.source:
            sources = [f"source_{i}" for i in self.source]
            index = index[index["Metadata_Source"].isin(sources)]   

        if self.batches:
            index = index[index["Metadata_Batch"].isin(self.batches)]
            
        return index


    def get_embedding_dataset(self, features, labels, metadata, batch_size, is_train):
            g = torch.Generator()
            g.manual_seed(self.seed)

            shuffle = is_train

            features_dataset = EmbeddingDataset(features, labels, metadata)
            features_dataloader = DataLoader(features_dataset, 
                                                num_workers=self.workers,
                                                batch_size=batch_size,
                                                worker_init_fn=seed_worker,
                                                pin_memory=True,
                                                persistent_workers=True,
                                                prefetch_factor=8,
                                                shuffle=shuffle
                                                )
            return features_dataloader
    

    def get_cellpainting_dataset(self, filelist, encoded_labels, metadata, preprocess_fn, img_path, is_train, rotation=False, batch_sampler=None, datasetclass=CellPainting, negcons_df=None, pair_variable=None, batch_size=None):
        dataset = datasetclass(
            img_path,
            filelist,
            encoded_labels,
            metadata,
            transforms=preprocess_fn,
            channel_subset=self.channels,
            negcons_df=negcons_df,
            pair_variable=pair_variable,
            rotation=rotation
            )
             
        shuffle = is_train

        if not batch_size:
            batch_size = self.batch_size

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
                persistent_workers=True,
               
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
    
    

    def get_per_domain_dataloaders(self, X_train, y_train, metadata_train, is_train, train_stats=False, batch_size=None, metadata_neg=None):
        normalize = None if not train_stats else self.normalize

        if not train_stats or "mean" in train_stats:
            preprocess_train = _transform(self.train_res, self.test_res, is_train=is_train, normalize=normalize, preprocess=self.preprocess, stats=train_stats)

        unique_domains = X_train[self.metadata_name].unique().tolist()
        unique_labels = self.metadata_encoder.transform(unique_domains)

        domain_loaders = {}

        for d, l in zip(unique_domains, unique_labels):
            X = X_train[X_train[self.metadata_name] == d]
            y = y_train[X.index]
            metadata = metadata_train[X.index]

            if train_stats and not "mean" in train_stats: 
                normalize = None if not train_stats else self.normalize
                preprocess_train = _transform(self.train_res, self.test_res, is_train=is_train, normalize=normalize, preprocess=self.preprocess, stats=train_stats[l])

            if self.label_shift:
                batch_sampler = LabelShiftGroupBatchSampler(metadata, y, self.total_size, self.batch_size_test, self.groups_per_batch_eval, eval=True, shift=self.shift, alpha=self.alpha, imb_ratio=self.imb_ratio, negative_groups=metadata_neg, neg_batch_size=self.neg_batch_size)
            else:
                batch_sampler = None

            domain_loader = self.get_cellpainting_dataset(X, y, metadata, preprocess_train, X["img_path"].tolist(), is_train=is_train, batch_size=batch_size, batch_sampler=batch_sampler)

            domain_loaders[l] = domain_loader

        return domain_loaders
        

    def get_groups(self, X, group_name):
        encoder = LabelEncoder()
        return encoder.fit_transform(X[group_name])







