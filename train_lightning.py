import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from lightning_model import LightningCenterNet
from lightning_datamodule import CenterNetDataModule
import json
from pycocotools.coco import COCO
import random
import numpy as np
from data_utils  import xml_to_coco_json

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # Set seed for reproducibility
    seed_everything(args.seed)
    
    # Load classes from COCO annotation file
    if os.path.exists(args.coco_annotation_path):
        cocoGt = COCO(args.coco_annotation_path)
        classes = []
        for (i, v) in cocoGt.cats.items():
            classes.append(v["name"])
        
        # Save classes to a file
        with open("classes.txt", "w") as f:
            for cls in classes:
                f.write(f"{cls}\n")
    else:
        val_outputs = xml_to_coco_json(os.path.join(args.data_dir,"val_images"), 'val_output_coco.json')
        cocoGt = COCO(args.coco_annotation_path)
        classes = []
        for (i, v) in cocoGt.cats.items():
            classes.append(v["name"])
        
        # Save classes to a file
        with open("classes.txt", "w") as f:
            for cls in classes:
                f.write(f"{cls}\n")
        # Try to load classes from classes.txt
        if os.path.exists("classes.txt"):
            with open("classes.txt", "r") as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            raise FileNotFoundError("Neither COCO annotation file nor classes.txt found")
    
    print(f"Classes: {classes}")
    
    # Initialize data module
    data_module = CenterNetDataModule(
        data_dir=args.data_dir,
        input_shape=(args.input_height, args.input_width),
        classes=classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stride=args.stride,
        seed=args.seed
    )
    
    # Initialize model
    model = LightningCenterNet(
        num_classes=len(classes),
        input_shape=(args.input_height, args.input_width),
        batch_size=args.batch_size,
        stride=args.stride,
        lr=args.learning_rate,
        min_lr=args.min_learning_rate,
        weight_decay=args.weight_decay,
        lr_decay_type=args.lr_scheduler,
        epochs=args.max_epochs,
        coco_gt_path=args.coco_annotation_path,
        val_data_path=args.data_dir,
        classes=classes,
        ciou_weight=args.ciou_weight,
        eval_interval = args.eval_interval
    )
    
    # Load pretrained weights if specified
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        # Load weights with compatibility handling
        pretrained_dict = torch.load(args.pretrained_weights, map_location='cpu')
        model_dict = model.model.state_dict()
        
        # Filter out incompatible keys
        compatible_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                compatible_dict[k] = v
        
        # Update model weights
        model_dict.update(compatible_dict)
        model.model.load_state_dict(model_dict)
        print(f"Loaded {len(compatible_dict)}/{len(model_dict)} layers from pretrained weights")
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='centernet-epoch{epoch:02d}-val_loss{val_loss:.4f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min'
    )
    
    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else None,
        precision='16-mixed' if args.fp16 else 32,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=args.val_check_interval
    )
    
    # Train model
    if args.resume_from_checkpoint != None:
        print(f"Resuming training from checkpoint {args.resume_from_checkpoint}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from_checkpoint)
    else:
        trainer.fit(model, data_module)
    
    # Print best model path
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CenterNet with PyTorch Lightning")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='E:/derpet_v4_label_tf',
                        help='Path to dataset directory')
    parser.add_argument('--coco_annotation_path', type=str, default='val_output_coco.json',
                        help='Path to COCO annotation file')
    
    # Model parameters
    parser.add_argument('--input_height', type=int, default=512, help='Input image height')
    parser.add_argument('--input_width', type=int, default=512, help='Input image width')
    parser.add_argument('--stride', type=int, default=4, help='Model stride')
    parser.add_argument('--ciou_weight', type=float, default=5.0, help='Weight for CIoU loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=5e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='yolox_cos', choices=['cos','yolox_cos','step'],
                        help='Learning rate scheduler type')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--fp16', default=True,action='store_true', help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=11, help='Random seed')
    
    # Checkpointing and logging
    parser.add_argument('--pretrained_weights', type=str, default='mbv4_cmini_0131map.pth', help='Path to pretrained weights')
    parser.add_argument('--log_dir', type=str, default='lightning_logs', help='Directory for logs')
    parser.add_argument('--experiment_name', type=str, default='centernet', help='Experiment name')
    parser.add_argument('--val_check_interval', type=int, default=1, help='Validation check interval (epochs)')
    parser.add_argument('--eval_interval', type=int, default=5,help='COCO evaluation interval (epochs)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume training from')
    
    # Hardware
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    
    args = parser.parse_args()
    main(args)