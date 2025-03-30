import torch
import torch.nn as nn
import pytorch_lightning as pl
from loss import focal_loss, ciou_loss,get_lr_scheduler,set_optimizer_lr,get_lr,iou_aware_loss,reg_l1_loss
from mbv4_timm import CenterNet
from utils_bbox import decode_bbox, postprocess
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from typing import Dict, List, Tuple, Optional, Any
import cv2
import glob 
from torch.optim.lr_scheduler import LambdaLR


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
        self.model = CenterNet(num_classes)
        
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
        # Only run COCO evaluation on main process
        current_epoch = self.current_epoch
        # Check if we should run evaluation this epoch
        should_evaluate = (current_epoch % self.eval_interval == 0) or (current_epoch == self.trainer.max_epochs - 1)
        if self.trainer.is_global_zero and self.cocoGt and should_evaluate:
            # Save model temporarily for evaluation
            temp_path = "temp_model_for_eval.pth"
            torch.save(self.model.state_dict(), temp_path)
            
            # Run COCO evaluation
            mean_ap = self.evaluate_coco(temp_path)
            
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
            if os.path.exists(temp_path):
                os.remove(temp_path)
        elif self.trainer.is_global_zero and self.cocoGt:
            pass
            #print(f"Skipping COCO evaluation at epoch {current_epoch} (will evaluate every {self.eval_interval} epochs)")
    
    def evaluate_coco(self, model_path):
        """Run COCO evaluation on the model"""
        if not self.cocoGt or not self.classes:
            return 0.0

        if self.current_epoch == 0:
            return 0.0
        
        # Print some info about ground truth annotations
        print(f"COCO GT info: {len(self.cocoGt.imgs)} images, {len(self.cocoGt.anns)} annotations")
        print(f"COCO categories: {self.cocoGt.cats}")
            
        # Prepare for evaluation
        folder = self.val_data_path
        print('folder:', folder)
        print('cocogt path:',self.coco_gt_path)
        val_images_folder = os.path.join(folder, "val_images")
        print('val images folder:',val_images_folder)
        # Get validation images
        val_images = []
        for ext in ["*.jpg", "*.png", "*.JPG"]:
            val_images.extend(glob.glob(os.path.join(val_images_folder, ext)))
        
        print(f"Found {len(val_images)} validation images")
        if len(val_images) == 0:
            print(f"No validation images found in {val_images_folder}")
            return 0.0
        
        # Run inference on validation images
        self.model.eval()
        results = []
        
        #for image_path in val_images:
        for i in self.cocoGt.dataset["images"]:
            try:
                image_id = i["id"]
                image_path = os.path.join(val_images_folder, i["file_name"])
                # Check if this image_id exists in the COCO ground truth
                if image_id not in self.cocoGt.imgs:
                    print(f"Warning: Image ID {image_id} not found in COCO annotations")
                
                # Read and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image {image_path}")
                    continue
                    
                if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
                    image = image # or pass
                else:
                    image = image.convert('RGB')
                
                # Preprocess image
                image_shape = np.array(image.shape[:2])
                image_data = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_CUBIC)
                image_data = image_data.astype('float32') / 255.0
                image_data = (image_data - np.array([0.40789655, 0.44719303, 0.47026116])) / np.array([0.2886383, 0.27408165, 0.27809834])
                image_data = np.transpose(image_data, (2, 0, 1))[None]
                
                # Run inference
                with torch.no_grad():
                    input_tensor = torch.from_numpy(image_data).float().to(self.device)
                    hm, wh, offset, iou = self.model(input_tensor)
                    
                    # Decode predictions
                    try:
                        outputs = decode_bbox(hm,wh,offset,iou,confidence=0.05)
                        
                        # Check if outputs is empty
                        if not outputs or len(outputs[0]) == 0:
                            print(f"No detections for image {image_id}")
                            continue
                            
                        results_boxes = postprocess(outputs,True,image_shape,self.input_shape, False, 0.2) 
                        
                        # Format results for COCO
                        for box in results_boxes[0]:
                            if len(box) < 6:  # Ensure box has all required values
                                continue
                                
                            y1, x1, y2, x2, conf, cls_id = box
                            
                            # Ensure box coordinates are valid
                            if x2 <= x1 or y2 <= y1:
                                continue
                                
                            # Ensure class_id is valid
                            class_id = int(cls_id) + 1  # COCO categories start from 1
                            if class_id not in self.cocoGt.cats:
                                print(f"Warning: Class ID {class_id} not in COCO categories")
                                continue
                                
                            results.append({
                                'image_id': image_id,
                                'category_id': class_id,
                                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                'score': float(conf)
                            })
                    except Exception as e:
                        print(f"Error in detection for image {image_path}: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
        
        # Save results to file
        with open('detection_results.json', 'w') as f:
            json.dump(results, f)
        
        print(f"Generated {len(results)} detections across {len(val_images)} images")
        
        # Check if we have any detections
        if len(results) == 0:
            print("No detections found in validation images. Cannot perform COCO evaluation.")
            return 0.0
        
        # Evaluate with COCO API
        try:
            # First validate that our results format is correct
            #for r in results[:5]:  # Print first few results for debugging
            #    print(f"Sample result: {r}")
                
            # Load results into COCO API
            cocoDt = self.cocoGt.loadRes('detection_results.json')
            
            # Make sure image IDs match between GT and detections
            gt_img_ids = set(self.cocoGt.getImgIds())
            dt_img_ids = set(cocoDt.getImgIds())
            common_img_ids = gt_img_ids.intersection(dt_img_ids)
            
            print(f"GT has {len(gt_img_ids)} images, DT has {len(dt_img_ids)} images")
            print(f"Common images: {len(common_img_ids)}")
            
            if len(common_img_ids) == 0:
                print("No common images between ground truth and detections!")
                return 0.0
                
            # Run evaluation
            cocoEval = COCOeval(self.cocoGt, cocoDt, 'bbox')
            
            # Optional: restrict evaluation to only images with detections
            # cocoEval.params.imgIds = list(common_img_ids)
            
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
            # Check if stats is available and has values
            if hasattr(cocoEval, 'stats') and len(cocoEval.stats) > 0:
                mean_ap = cocoEval.stats[0]  # mAP at IoU thresholds from .50 to .95
                print(f"mAP: {mean_ap:.4f}")
                return mean_ap
            else:
                print("COCO evaluation completed but stats are not available")
                return 0.0
        except Exception as e:
            print(f"COCO evaluation error: {e}")
            # Print more detailed error information
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
    
    def on_train_epoch_start(self):
        # Only apply manual LR scheduling if using yolox_cos
        if self.lr_decay_type == "yolox_cos":
            # Get current epoch
            current_epoch = self.current_epoch
            # Calculate the new learning rate based on the current epoch
            set_optimizer_lr(self.optimizers(), self.lr_scheduler_func, current_epoch)
            current_lr = get_lr(self.optimizers())
            # Log the learning rate
            self.log('learning_rate', current_lr, prog_bar=True)
