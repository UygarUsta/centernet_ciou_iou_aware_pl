import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset # Added Dataset
from dataloader import CenternetDataset, centernet_dataset_collate
from glob import glob
from functools import partial
import random
import numpy as np

class RepeatDataset(Dataset):
    def __init__(self, dataset, times=1):
        self.dataset = dataset
        self.times = times

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

    def __len__(self):
        return len(self.dataset) * self.times
    
    # Helper to easily access the underlying dataset attributes (like classes)
    def __getattr__(self, name):
        return getattr(self.dataset, name)


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class CenterNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        input_shape: tuple = (512, 512),
        classes: list = None,
        batch_size: int = 16,
        num_workers: int = 8,
        stride: int = 4,
        use_ttf: bool = False,
        seed: int = 11,
        mosaic=True, 
        mixup=True,
        repeats: int = 1, # New parameter: defaults to 1 (no repeat)
        train_annotation_path: str = None,
        val_annotation_path: str = None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.classes = classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.stride = stride
        self.use_ttf = use_ttf
        self.seed = seed
        self.mosaic = mosaic
        self.mixup = mixup
        self.repeats = repeats # Store repeats

        self.train_annotation_path = train_annotation_path
        self.val_annotation_path = val_annotation_path
        
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        train_images = []
        val_images = []
        train_annotations = None
        val_annotations = None

        # Load Train Data logic
        if self.train_annotation_path and os.path.exists(self.train_annotation_path):
            from data_utils import load_coco_data
            train_img_root = os.path.join(self.data_dir, "train")
            train_images, train_annotations = load_coco_data(self.train_annotation_path, train_img_root, self.classes)
        else:
            for ext in ["*.jpg", "*.png", "*.JPG"]:
                train_images.extend(glob(os.path.join(self.data_dir, "train", ext)))
            train_images = sorted(train_images)

        # Load Val Data logic
        if self.val_annotation_path and os.path.exists(self.val_annotation_path):
            from data_utils import load_coco_data
            val_img_root = os.path.join(self.data_dir, "valid")
            val_images, val_annotations = load_coco_data(self.val_annotation_path, val_img_root, self.classes)
        else:
            for ext in ["*.jpg", "*.png", "*.JPG"]:
                val_images.extend(glob(os.path.join(self.data_dir, "valid", ext)))
            val_images = sorted(val_images)
        
        # Initialize the base dataset
        base_train_dataset = CenternetDataset(
            train_images,
            self.input_shape,
            self.classes,
            len(self.classes),
            train=True,
            stride=self.stride,
            mosaic=self.mosaic,
            mixup=self.mixup,
            coco_annotations=train_annotations
        )

        # --- Apply Repeat Logic here ---
        if self.repeats > 1:
            self.train_dataset = RepeatDataset(base_train_dataset, times=self.repeats)
        else:
            self.train_dataset = base_train_dataset
        
        self.val_dataset = CenternetDataset(
            val_images,
            self.input_shape,
            self.classes,
            len(self.classes),
            train=False,
            stride=self.stride,
            coco_annotations=val_annotations
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle still works perfectly on the repeated indices
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=centernet_dataset_collate,
            worker_init_fn=partial(worker_init_fn, rank=0, seed=self.seed),
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=centernet_dataset_collate,
            worker_init_fn=partial(worker_init_fn, rank=0, seed=self.seed),
            persistent_workers=True
        )
    
    def disable_augmentations(self):
        # We need to check if it's wrapped or not
        target = self.train_dataset
        if isinstance(target, RepeatDataset):
            target = target.dataset
            
        if target is not None:
            target.mosaic = False
            target.mixup = False