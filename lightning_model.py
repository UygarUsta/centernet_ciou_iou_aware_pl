import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import focal_loss, ciou_loss,get_lr_scheduler,set_optimizer_lr,get_lr,iou_aware_loss,reg_l1_loss
from mbv4_timm import CenterNet
from hardnet import get_pose_net
from lightning_datamodule import CenterNetDataModule
from utils_bbox import decode_bbox, postprocess
import numpy as np
import json
import faster_coco_eval
# This single line replaces pycocotools with faster-coco-eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from typing import Dict, List, Tuple, Optional, Any
import cv2
import glob 
from torch.optim.lr_scheduler import LambdaLR
from datetime import datetime
import csv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.distributed as dist
from PIL import Image 

class COCOEvalDataset(Dataset):
    def __init__(self, coco, img_dir, input_shape):
        self.coco = coco
        self.img_dir = img_dir
        self.input_shape = input_shape
        # Get all image IDs
        self.img_ids = sorted(coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_dir, file_name)
        
        # Load and preprocess
        image = cv2.imread(image_path)
        if image is None:
            # Return dummy if failed (handled in loop)
            return torch.zeros((3, *self.input_shape)), img_id, 0, 0
            
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Ensure RGB

        # Save original shape for restoration
        orig_h, orig_w = image.shape[:2]
        
        # Resize and Normalize
        image_data = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_CUBIC)
        image_data = image_data.astype('float32') / 255.0
        # Mean/Std normalization
        image_data = (image_data - np.array([0.40789655, 0.44719303, 0.47026116])) / np.array([0.2886383, 0.27408165, 0.27809834])
        image_data = np.transpose(image_data, (2, 0, 1))
        
        return torch.from_numpy(image_data).float(), img_id, orig_h, orig_w
    


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

def preprocess_input(image):
    image   = np.array(image,dtype = np.float32)[:, :, ::-1]
    mean    = [0.40789655, 0.44719303, 0.47026116]
    std     = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std
    
class COCOEvalDataset2(Dataset):
    def __init__(self, coco, img_dir, input_shape):
        self.coco = coco
        self.img_dir = img_dir
        self.input_shape = input_shape
        # Get all image IDs
        self.img_ids = sorted(coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_dir, file_name)
        
        image   = Image.open(image_path)
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = self.input_shape


        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        #---------------------------------#
        #   将图像多余的部分加上灰条
        #---------------------------------#
        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data  = np.array(new_image, np.float32)

        image_data = np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1))

        image = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)

        if image is None:
            # Return dummy if failed (handled in loop)
            return torch.zeros((3, *self.input_shape)), img_id, 0, 0
            
        
        return image, img_id, ih, iw  #orig_h, orig_w


def decode_boxes_for_ciou(hm,offset, wh, batch_regs, batch_whs, batch_reg_masks, device_type='cuda'):
    """
    Ground truth ve tahmin edilen kutuları CIoU loss için decode eder.
    
    Args:
        hm (torch.Tensor): Heatmap değerleri [B, H, W,C]
        offset (torch.Tensor): Merkez offset değerleri [B, 2, H, W]
        wh (torch.Tensor): Genişlik ve yükseklik değerleri [B, 2, H, W]
        batch_regs (torch.Tensor): Ground truth regresyon değerleri [B, H, W, 2]
        batch_whs (torch.Tensor): Ground truth genişlik ve yükseklik değerleri [B, H, W, 2]
        batch_reg_masks (torch.Tensor): Regresyon maskeleri [B, H, W]
        grid_h (int): Grid yüksekliği
        grid_w (int): Grid genişliği
        device_type (torch.device): İşlem yapılacak cihaz
        
    Returns:
        tuple: (pred_bboxes, gt_bboxes, mask) - Tahmin edilen kutular, ground truth kutular ve geçerli maske
    """

    batch_size, _, grid_h, grid_w = hm.size()
    
    
    # Create grid using meshgrid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(grid_h, device=device_type),
        torch.arange(grid_w, device=device_type)
    )
    grid_x = grid_x.float().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    grid_y = grid_y.float().unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Compute center points
    ct_x = grid_x + offset[:, 0:1, :, :]
    ct_y = grid_y + offset[:, 1:2, :, :]
    ct = torch.cat([ct_x, ct_y], dim=1)  # [B, 2, H, W]

    # Compute bounding boxes
    x1 = ct[:, 0, :, :] - wh[:, 0, :, :] / 2
    y1 = ct[:, 1, :, :] - wh[:, 1, :, :] / 2
    x2 = ct[:, 0, :, :] + wh[:, 0, :, :] / 2
    y2 = ct[:, 1, :, :] + wh[:, 1, :, :] / 2
    pred_bboxes = torch.stack([x1, y1, x2, y2], dim=1)  # [B, 4, H, W]

    # Permute batch_regs and batch_whs to match expected dimensions
    batch_regs = batch_regs.permute(0, 3, 1, 2)
    batch_whs = batch_whs.permute(0, 3, 1, 2)
    
    # Ground truth bounding boxes
    gt_wh = batch_whs  # [B, 2, H, W]
    gt_ct_x = grid_x + batch_regs[:, 0:1, :, :]
    gt_ct_y = grid_y + batch_regs[:, 1:2, :, :]
    gt_x1 = gt_ct_x - gt_wh[:, 0:1, :, :] / 2
    gt_y1 = gt_ct_y - gt_wh[:, 1:2, :, :] / 2
    gt_x2 = gt_ct_x + gt_wh[:, 0:1, :, :] / 2
    gt_y2 = gt_ct_y + gt_wh[:, 1:2, :, :] / 2
    gt_bboxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)  # [B, 4, H, W]
    gt_bboxes = torch.squeeze(gt_bboxes, 2)

    # Reshape for CIoU loss
    pred_bboxes = pred_bboxes.permute(0, 2, 3, 1).reshape(-1, 4)  # [B*H*W, 4]
    gt_bboxes = gt_bboxes.permute(0, 2, 3, 1).reshape(-1, 4)      # [B*H*W, 4]
    mask = batch_reg_masks.view(-1) > 0
    
    return pred_bboxes, gt_bboxes, mask


def calc_iou(pred_bboxes,gt_bboxes):
    with torch.no_grad():
        inter_x1 = torch.max(pred_bboxes[:,0], gt_bboxes[:,0])
        inter_y1 = torch.max(pred_bboxes[:,1], gt_bboxes[:,1])
        inter_x2 = torch.min(pred_bboxes[:,2], gt_bboxes[:,2])
        inter_y2 = torch.min(pred_bboxes[:,3], gt_bboxes[:,3])

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h

        area_pred = (pred_bboxes[:,2]-pred_bboxes[:,0]).clamp(min=0)*(pred_bboxes[:,3]-pred_bboxes[:,1]).clamp(min=0)
        area_gt = (gt_bboxes[:,2]-gt_bboxes[:,0]).clamp(min=0)*(gt_bboxes[:,3]-gt_bboxes[:,1]).clamp(min=0)
        union = area_pred + area_gt - inter_area + 1e-6
        actual_iou = inter_area / union  # shape: [N]
    return actual_iou

class LightningCenterNet(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        input_shape: Tuple[int, int] = (512, 512),
        batch_size : int = 16,
        stride: int = 4,
        lr: float = 5e-4,
        min_lr: float = 5e-6,
        weight_decay: float = 0,
        lr_decay_type: str = "yolox_cos",
        epochs: int = 100,
        coco_gt_path: Optional[str] = None,
        val_data_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        ciou_weight: float = 5.0,
        eval_interval: int = 5
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        #self.model = CenterNet(num_classes)
        
        self.model = get_pose_net(68,{"hm":num_classes,"wh":2,"offset":2,"iou":1})
        # Parameters
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.stride = stride
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.lr_decay_type = lr_decay_type
        self.epochs = epochs
        self.ciou_weight = ciou_weight
        self.eval_interval = eval_interval
        
        # COCO evaluation
        self.coco_gt_path = coco_gt_path
        self.val_data_path = val_data_path
        self.classes = classes
        if self.coco_gt_path and os.path.exists(self.coco_gt_path):
            self.cocoGt = COCO(self.coco_gt_path)
        else:
            self.cocoGt = None
        
        # Best mAP tracking
        self.best_map = 0.0
 
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        
        # Forward pass
        hm,wh,offset,iou = self(batch_images)
        
        # Classification loss (focal loss)
        c_loss = focal_loss(hm, batch_hms)

        #Offset Loss (L1)
        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        
        # Regression loss (CIoU loss)
        pred_boxes, gt_boxes, mask = decode_boxes_for_ciou(hm, offset, wh, batch_regs, batch_whs, batch_reg_masks)
        loss_ciou = ciou_loss(pred_boxes, gt_boxes, mask)

        #IoU Loss
        actual_iou = calc_iou(pred_boxes, gt_boxes)
        iou_pred_flat = iou.view(-1)
        iou_aware = iou_aware_loss(iou_pred_flat,actual_iou,mask)
        
        # Total loss
        loss = c_loss + loss_ciou * self.ciou_weight + iou_aware + off_loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_c_loss', c_loss, prog_bar=False)
        self.log('train_ciou_loss', loss_ciou, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        
        # Forward pass
        hm,wh,offset,iou = self(batch_images)
        
        # Classification loss (focal loss)
        c_loss = focal_loss(hm, batch_hms)

        #Offset Loss (L1)
        off_loss        = reg_l1_loss(offset, batch_regs, batch_reg_masks)
        
        # Regression loss (CIoU loss)
        pred_boxes, gt_boxes, mask = decode_boxes_for_ciou(hm, offset, wh, batch_regs, batch_whs, batch_reg_masks)
        loss_ciou = ciou_loss(pred_boxes, gt_boxes, mask)

        #IoU Loss
        actual_iou = calc_iou(pred_boxes, gt_boxes)
        iou_pred_flat = iou.view(-1)
        iou_aware = iou_aware_loss(iou_pred_flat,actual_iou,mask)
        
        # Total loss
        loss = c_loss + loss_ciou * self.ciou_weight + iou_aware + off_loss
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_c_loss', c_loss, prog_bar=False, sync_dist=True)
        self.log('val_ciou_loss', loss_ciou, prog_bar=False, sync_dist=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        mean_ap = 0.0
        # Only run COCO evaluation on main process
        current_epoch = self.current_epoch + 1
        print("Current Epoch is:",current_epoch)
        # Check if we should run evaluation this epoch
        should_evaluate = (current_epoch % self.eval_interval == 0) or (current_epoch == self.trainer.max_epochs - 1) 
        if self.trainer.is_global_zero and self.cocoGt and should_evaluate:
            # Save model temporarily for evaluation
            temp_path = "temp_model_for_eval.pth"
            torch.save(self.model.state_dict(), temp_path)
            
            # Run COCO evaluation
            coco_eval = self.evaluate_coco(temp_path)
            if isinstance(coco_eval,COCOeval):
                mean_ap = coco_eval.stats[0]  # This is the AP@[IoU=0.50:0.95]
            else:
                mean_ap = 0.0

            # Log mAP to CSV
            if self.current_epoch > 0:
                self.log_map_to_csv(coco_eval)
            
            # Log mAP
            self.log('val_mAP', mean_ap, prog_bar=True)

            
            
            # Save best model
            if mean_ap > self.best_map:
                self.best_map = mean_ap
                self.log('best_mAP', self.best_map)
                # Ensure the checkpoint directory exists
                if hasattr(self.trainer, 'checkpoint_callback') and hasattr(self.trainer.checkpoint_callback, 'dirpath'):
                    checkpoint_dir = self.trainer.checkpoint_callback.dirpath
                    if checkpoint_dir is not None:
                        # Make sure the directory exists
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        # Save the best model
                        best_model_path = os.path.join(checkpoint_dir, f"best_model_mAP_{mean_ap:.4f}.pth")
                        torch.save(self.model.state_dict(), best_model_path)
                        
                        # Update the best model path in the checkpoint callback
                        self.trainer.checkpoint_callback.best_model_path = best_model_path
                    else:
                        print("Warning: Checkpoint directory not available yet, skipping best model save")
                else:
                    print("Warning: Checkpoint callback not available, skipping best model save")
            
            # Clean up
            # if os.path.exists(temp_path):
            #     os.remove(temp_path)

        if should_evaluate and self.trainer.num_devices > 1:
            
            dist.broadcast(torch.tensor(mean_ap, device=self.device), src=0)
            
        # 3. Log: Both ranks must log to prevent hanging
        if should_evaluate:
            self.log('val_mAP', mean_ap, prog_bar=True, sync_dist=False)

        elif self.trainer.is_global_zero and self.cocoGt:
            #pass
            print(f"Skipping COCO evaluation at epoch {current_epoch} (will evaluate every {self.eval_interval} epochs)")

    def log_map_to_csv(self, coco_eval):
        """Log all COCO evaluation metrics to CSV file"""
        # Create the CSV file path in the same directory as checkpoints
        csv_path = os.path.join(self.logger.log_dir, "map_results.csv")
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(csv_path)
        
        # Get current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Define all the metrics we want to save
        fieldnames = [
            'epoch', 'date_time', 'learning_rate',
            'AP_all', 'AP_50', 'AP_75', 'AP_small', 'AP_medium', 'AP_large',
            'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large'
        ]
        
        # Extract all metrics from coco_eval
        metrics = {
            'AP_all': coco_eval.stats[0],
            'AP_50': coco_eval.stats[1],
            'AP_75': coco_eval.stats[2],
            'AP_small': coco_eval.stats[3],
            'AP_medium': coco_eval.stats[4],
            'AP_large': coco_eval.stats[5],
            'AR_1': coco_eval.stats[6],
            'AR_10': coco_eval.stats[7],
            'AR_100': coco_eval.stats[8],
            'AR_small': coco_eval.stats[9],
            'AR_medium': coco_eval.stats[10],
            'AR_large': coco_eval.stats[11]
        }
        
        # Get current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else 0
        
        # Open file in append mode
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writeheader()
            
            # Create row with all metrics
            row = {
                'epoch': self.current_epoch,
                'date_time': current_time,
                'learning_rate': f"{current_lr:.8f}"
            }
            
            # Add all metrics to the row
            for metric_name, metric_value in metrics.items():
                row[metric_name] = f"{metric_value:.6f}"
            
            # Write the row
            writer.writerow(row)
        
        print(f"All COCO metrics logged to {csv_path}")
    
    
    
    def evaluate_coco(self, model_path):
        """High-performance COCO evaluation using Batched DataLoader"""
        if not self.cocoGt or not self.classes:
            return 0.0

        if self.current_epoch == 0:
            return 0.0

        # Mapping: Class Name -> COCO ID
        cat_name_to_id = {cat['name']: cat['id'] for cat in self.cocoGt.cats.values()}
        
        val_images_folder = os.path.join(self.val_data_path, "valid")
        
        # 1. Create the optimized DataLoader
        eval_dataset = COCOEvalDataset2(self.cocoGt, val_images_folder, self.input_shape) #Letterbox applied
        
        # Use a reasonable batch size (e.g., 32) and workers (e.g., 4 or 8)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=16,       # Increase this if you have GPU memory
            shuffle=False, 
            num_workers=4,       # Parallel loading
            pin_memory=True
        )

        print(f"Starting evaluation on {len(eval_dataset)} images...")
        
        self.model.eval()
        results = []
        
        # 2. Batched Inference Loop
        with torch.no_grad():
            for batch_imgs, batch_ids, batch_orig_h, batch_orig_w in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to GPU
                batch_imgs = batch_imgs.to(self.device)
                
                # Inference
                hm, wh, offset, iou = self.model(batch_imgs)
                
                # Decode bounding boxes
                # Note: decode_bbox processes the whole batch at once
                outputs = decode_bbox(hm, wh, offset, iou, confidence=0.05)
                
                # Process each image in the batch
                for i in range(len(outputs)):
                    img_id = int(batch_ids[i])
                    orig_h = int(batch_orig_h[i])
                    orig_w = int(batch_orig_w[i])
                    
                    # Skip failed images (dummy returns)
                    if orig_h == 0: continue
                    
                    output = outputs[i]
                    if output is None or len(output) == 0:
                        continue
                        
                    # Post-process (Rescale boxes to original image size)
                    # We pass single-item lists to match expected signature
                    image_shape = np.array([orig_h, orig_w])
                    results_boxes = postprocess([output], False, image_shape, self.input_shape, True, 0.2) #Letterbox set to True, NMS set to false (higher map when set to false)
                    
                    # Format for JSON
                    for box in results_boxes[0]:
                        if len(box) < 6: continue
                        y1, x1, y2, x2, conf, cls_id = box
                        
                        if x2 <= x1 or y2 <= y1: continue
                        
                        # ID Mapping Logic
                        pred_class_idx = int(cls_id)
                        if pred_class_idx >= len(self.classes): continue
                        
                        pred_class_name = self.classes[pred_class_idx]
                        if pred_class_name not in cat_name_to_id: continue
                        category_id = cat_name_to_id[pred_class_name]
                        
                        results.append({
                            'image_id': img_id,
                            'category_id': category_id,
                            'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            'score': float(conf)
                        })

        # 3. Standard COCO Eval (same as before)
        if len(results) == 0:
            print("No detections found.")
            return 0.0
            
        # Save results
        with open('detection_results.json', 'w') as f:
            json.dump(results, f)
            
        try:
            cocoDt = self.cocoGt.loadRes('detection_results.json')
            cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return cocoEval
        except Exception as e:
            print(f"COCO evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def configure_optimizers(self):

        Init_lr = self.lr #5e-4
        Min_lr  = Init_lr * 0.01
        momentum = 0.9
        nbs             = 64
        lr_limit_max    = 5e-4 
        lr_limit_min    = 2.5e-4 
        Init_lr_fit     = min(max(self.batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(self.batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=Init_lr_fit,
            betas = (momentum, 0.999),
            weight_decay=self.weight_decay
        )


        #total_iters = self.trainer.estimated_stepping_batches
        
        # Learning rate scheduler
        if self.lr_decay_type == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.epochs,
                eta_min=self.min_lr
            )
        elif self.lr_decay_type == "yolox_cos":
            # Get the learning rate scheduler functions
            lr_scheduler_func = get_lr_scheduler(
                lr_decay_type='cos',
                lr=Init_lr_fit,
                min_lr=Min_lr_fit,
                total_iters=self.epochs  #total_iters
            )
            #scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_scheduler_func(epoch))
            self.lr_scheduler_func = lr_scheduler_func
            
            return optimizer
            
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.epochs // 10,
                gamma=0.1
            )
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    # def on_train_epoch_start(self):
    #     # Only apply manual LR scheduling if using yolox_cos
    #     if self.lr_decay_type == "yolox_cos":
    #         # Get current epoch
    #         current_epoch = self.current_epoch
    #         # Calculate the new learning rate based on the current epoch
    #         set_optimizer_lr(self.optimizers(), self.lr_scheduler_func, current_epoch)
    #         current_lr = get_lr(self.optimizers())
    #         # Log the learning rate
    #         self.log('learning_rate', current_lr, prog_bar=True)

    #     # Check if we're in the last 10 epochs
    #     if self.trainer.max_epochs - self.current_epoch <= 10:
    #         # Disable mosaic and mixup
    #         if self.current_epoch == self.trainer.max_epochs - 10:
    #             print(f"Epoch {self.current_epoch}: Disabling mosaic and mixup for final training")
    #         if isinstance(self.trainer.datamodule, CenterNetDataModule):
    #             self.trainer.datamodule.disable_augmentations()

    def on_train_epoch_start(self):
        # 1. Retrieve the repeat factor from your datamodule
        # Default to 1 if not found
        repeats = getattr(self.trainer.datamodule, "repeats", 1)
        
        # 2. Scale the "no augmentation" period
        # If repeats=5, then 10 // 5 = 2 epochs.
        # 2 epochs * 5 repeats = 10 actual passes (Same as original intent)
        no_aug_epochs = max(1, 10 // repeats) 

        # --- LR Scheduler Logic (Remains mostly the same) ---
        if self.lr_decay_type == "yolox_cos":
            current_epoch = self.current_epoch
            set_optimizer_lr(self.optimizers(), self.lr_scheduler_func, current_epoch)
            current_lr = get_lr(self.optimizers())
            self.log('learning_rate', current_lr, prog_bar=True)

        # --- Augmented Disable Logic (Updated) ---
        # Use the scaled 'no_aug_epochs' variable instead of hardcoded 10
        if self.trainer.max_epochs - self.current_epoch <= no_aug_epochs:
            
            # Check if we just entered this stage to print the message
            if self.current_epoch == self.trainer.max_epochs - no_aug_epochs:
                print(f"Epoch {self.current_epoch}: Disabling mosaic/mixup for final {no_aug_epochs} epochs "
                    f"(equivalent to {no_aug_epochs * repeats} original epochs)")
            
            if isinstance(self.trainer.datamodule, CenterNetDataModule):
                self.trainer.datamodule.disable_augmentations()

