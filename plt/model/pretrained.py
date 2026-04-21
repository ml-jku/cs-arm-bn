import os
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate

from plt.data.transforms import _transform


class PreTrainedARMBN():
    def __init__(self, path_dir='example/models/pretrained/', config='config/trainer/model/featurizer.yaml', data_root=".", device='cuda:0', **kwargs):
        self.path_dir = Path(path_dir)
        self.data_root = Path(data_root)
        self.checkpoints = sorted(glob.glob(os.path.join(path_dir, "*.pt")))
        self.models = [self._load(c, config) for c in self.checkpoints]
        self.transform = _transform(256, 256, is_train=False, normalize="dataset", preprocess="crop+flip", stats={"mean":[0.0913, 0.0880, 0.1122, 0.0765, 0.0598], "std":[0.1152, 0.1075, 0.1067, 0.0796, 0.0944]})

        assert self.models, "Pretrained checkpoints were not correctly loaded, plase provide a correct path_dir and config"


    def preprocess(self, input_df):
        "Process raw tiff files into numpy arrays"
        cols = [f"FileName_Orig{c}" for c in ["RNA", "ER", "AGP", "Mito", "DNA"]]
        base_paths = input_df["img_path"].apply(lambda p: str((self.data_root / p).resolve()) if not Path(p).is_absolute() else p )
        full_img_paths = input_df[cols].apply(lambda col: base_paths.str.cat(col, sep=os.sep))

        imglist = [row.tolist() for _, row in full_img_paths[cols].iterrows()]


        images = [self._read_img(i) for i in imglist]
        images = torch.stack(images, dim=0)

        return images
    
    def reset_bn_params(self, model):
        for nm, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = None
                m.reset_running_stats()


    def adapt(self, input_df):
        "Adapt models BN statistics given a list of tiff files"

        for model in self.models:
            model.train()
            self.reset_bn_params(model)

        images = self.preprocess(input_df)

        with torch.no_grad():
            for m in self.models:
                m(images)


    def predict(self, input_df):
        """Encode image given a list of tiff files"""   
             
        for model in self.models:
            model.eval()

        images = self.preprocess(input_df)

        all_logits = []

        with torch.no_grad():
            for m in self.models:
                logits = m(images)
                all_logits.append(logits)

        all_logits = torch.stack(all_logits)

        probs = torch.softmax(all_logits, dim=-1)  
        mean_probs = probs.mean(dim=0)     
        
        uncertainty = -(mean_probs * mean_probs.log()).sum(dim=-1)

        return mean_probs, uncertainty


    def _read_img(self, lst):
        assert len(lst) == 5, f"Input has {len(lst)} channels, it should have 5"

        images = [Image.open(f) for f in lst]
        images = [np.array(i.resize([512,512])) for i in images]

        thres = [self._illumination_threshold(i) for i in images]
        images = [self._sixteen_to_eight_bit(i, t) for i, t in zip(images, thres)]

        image = torch.from_numpy(np.stack(images, axis=0))
        image = self.transform[0](image)
    
        return image


    def _load(self, checkpoint, config):
        state_dict = torch.load(checkpoint, weights_only=True)

        assert os.path.exists(config)
        c = OmegaConf.load(config)

        model = instantiate(c)
        model.load_state_dict(state_dict)
        return model
    

    def _sixteen_to_eight_bit(self, arr, display_max, display_min=0):
        threshold_image = ((arr.astype(float) - display_min) * (arr > display_min))

        scaled_image = (threshold_image * (255 / (display_max - display_min)))
        scaled_image[scaled_image > 255] = 255

        scaled_image = scaled_image.astype(np.uint8)

        return scaled_image
    

    def _illumination_threshold(self, arr, perc=0.01):
        """ Return threshold value to not display a percentage of highest pixels"""

        perc = perc/100

        h = arr.shape[0]
        w = arr.shape[1]

        # find n pixels to delete
        total_pixels = h * w
        n_pixels = total_pixels * perc
        n_pixels = int(np.around(n_pixels))

        # find indexes of highest pixels
        flat_inds = np.argpartition(arr, -n_pixels, axis=None)[-n_pixels:]
        inds = np.array(np.unravel_index(flat_inds, arr.shape)).T

        max_values = [arr[i, j] for i, j in inds]

        threshold = min(max_values)

        return threshold