import math
import torch 
import numpy as np 
from plt.utils import split_by_groups


class GroupBatchSampler:
    def __init__(self, groups, batch_size, groups_per_batch, eval, n_samples_per_group=None, negative_groups=None, neg_batch_size=None, seed=1234):
        self.groups = torch.as_tensor(groups)
        self.batch_size = batch_size
        self.neg_batch_size = neg_batch_size
        self.eval = eval
        self.seed = seed

        self.group_indices, self.unique_groups, _ = split_by_groups(self.groups)

        self.groups_per_batch = min(groups_per_batch, len(self.unique_groups))

        self.negative_groups = torch.as_tensor(negative_groups) if negative_groups is not None else negative_groups

        if self.negative_groups is not None:
            self.neg_group_indices, self.neg_unique_groups, _ = split_by_groups(self.negative_groups)

        if not self.eval:
            self.num_batches = len(groups) // self.batch_size
        else:
            self.num_batches = sum([math.ceil(len(group_ids) / self.batch_size) for group_ids in self.group_indices.values()])

        self.n_samples_per_group = self.batch_size // self.groups_per_batch if not n_samples_per_group else n_samples_per_group


    def __iter__(self):
        batch_ids = []

        if not self.eval: 
            for i in range(self.num_batches):

                batch_groups = np.random.choice(self.unique_groups, self.groups_per_batch, replace=False)

                sample_ids = [np.random.choice(self.group_indices[g], 
                                                self.n_samples_per_group,
                                                replace=len(self.group_indices[g]) <= self.n_samples_per_group) 

                                for g in batch_groups]
                
                if self.negative_groups is not None:
                    neg_sample_ids = [np.random.choice(self.neg_group_indices[g], 
                                        self.neg_batch_size, 
                                        replace=len(self.neg_group_indices[g]) <= self.neg_batch_size) + len(self.groups)
                                        for g in batch_groups]
                    

                sample_ids = np.concatenate(sample_ids)

                if self.negative_groups is not None:
                    neg_sample_ids = np.concatenate(neg_sample_ids)
                    sample_ids = np.concatenate([sample_ids, neg_sample_ids])

                batch_ids.append(sample_ids)
        else:
            rng = np.random.default_rng(self.seed)

            for g, group_ids in self.group_indices.items():
                num_batches = math.ceil(len(group_ids) / (self.batch_size)) 

                group_ids_permuted = rng.permutation(group_ids)

                for i in range(num_batches):
                    end = min(len(group_ids), (i+1)* self.batch_size)
                    sample_ids = group_ids_permuted[i*self.batch_size:end]

                    if self.negative_groups is not None:
                        neg_sample_ids = np.random.choice(self.neg_group_indices[g], 
                                            self.neg_batch_size,          
                                            replace=len(self.neg_group_indices[g]) <= self.neg_batch_size) + len(self.groups)                      
                        sample_ids = np.concatenate([sample_ids, neg_sample_ids])

                    batch_ids.append(sample_ids)
            
        return iter(batch_ids)
    

    def __len__(self):
        return self.num_batches



class LabelShiftGroupBatchSampler(GroupBatchSampler):
    def __init__(self, groups, y, total_size, batch_size, groups_per_batch, eval, shift, seed=5678, alpha=None, imb_ratio=None, n_samples_per_group=None, negative_groups=None, neg_batch_size=None):
        super().__init__(groups, batch_size, groups_per_batch, eval, n_samples_per_group, negative_groups, neg_batch_size)
        self.shift = shift
        self.imb_ratio = imb_ratio
        self.y = torch.from_numpy(y)
        self.num_classes = len(torch.unique(self.y))

        if self.shift=="tail":
            self.class_probs = [1 * (self.imb_ratio ** (i / (self.num_classes - 1))) for i in range(self.num_classes)]
        elif self.shift=="dirichlet":
            self.alpha = alpha
            self.alpha = np.ones(self.num_classes) * alpha
            self.rng = np.random.default_rng(seed)

        self.total_size = total_size        
        self.class_indices, _, _ = split_by_groups(self.y)


    def __iter__(self):
        batch_ids = []

        if self.eval:
            for g in self.group_indices.keys():
                dirichlet_probs = self.rng.dirichlet(self.alpha)
                samples_per_class = list(self.rng.multinomial(self.total_size, dirichlet_probs))
                group_ids = self.group_indices[g]

                for i in range(self.num_classes):
                    sample_ids = []

                    for c in self.class_indices.keys():
                        k = samples_per_class[c]

                        class_ids = self.class_indices[c]
                        ids = torch.isin(group_ids, class_ids)
                        ids = group_ids[ids]

                        randids = torch.randperm(len(ids))
                        shifted_indices_perm = ids[randids]

                        end = min(len(shifted_indices_perm), k)

                        sample_class_ids = shifted_indices_perm[:end]
                        sample_ids.append(sample_class_ids)

                    sample_ids = np.concatenate(sample_ids)

                    if self.negative_groups is not None:
                        neg_sample_ids = np.random.choice(self.neg_group_indices[g], 
                                                self.neg_batch_size) + len(self.groups)
                                                    
                        sample_ids = np.concatenate([sample_ids, neg_sample_ids])
                    
                    samples_per_class = [samples_per_class[-1]] + samples_per_class[:-1]

                    batch_ids.append(sample_ids)

        return iter(batch_ids)

