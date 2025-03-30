import os
import glob
import json
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
from infer_utils import load_model
from utils_bbox import decode_bbox, postprocess
from tqdm import tqdm 

def evaluate_coco(model, device, val_data_path, coco_gt_path, input_shape=(512, 512), confidence_threshold=0.05, nms_threshold=0.2):
    """Run COCO evaluation on the model"""
    # Load COCO ground truth
    cocoGt = COCO(coco_gt_path)
    if not cocoGt:
        print("Failed to load COCO ground truth")
        return 0.0
    
    # Print some info about ground truth annotations
    print(f"COCO GT info: {len(cocoGt.imgs)} images, {len(cocoGt.anns)} annotations")
    print(f"COCO categories: {cocoGt.cats}")
        
    # Prepare for evaluation
    folder = val_data_path
    print('folder:', folder)
    print('cocogt path:', coco_gt_path)
    val_images_folder = os.path.join(folder, "val_images")
    print('val images folder:', val_images_folder)
    
    # Get validation images
    val_images = []
    for ext in ["*.jpg", "*.png", "*.JPG"]:
        val_images.extend(glob.glob(os.path.join(val_images_folder, ext)))
    
    print(f"Found {len(val_images)} validation images")
    if len(val_images) == 0:
        print(f"No validation images found in {val_images_folder}")
        return 0.0
    
    # Run inference on validation images
    model.eval()
    results = []
    
    for i in tqdm(cocoGt.dataset["images"]):
        try:
            image_id = i["id"]
            image_path = os.path.join(val_images_folder, i["file_name"])
            # Check if this image_id exists in the COCO ground truth
            if image_id not in cocoGt.imgs:
                print(f"Warning: Image ID {image_id} not found in COCO annotations")
            
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
                image = image  # or pass
            else:
                image = image.convert('RGB')
            
            # Preprocess image
            image_shape = np.array(image.shape[:2])
            image_data = cv2.resize(image, input_shape, interpolation=cv2.INTER_CUBIC)
            image_data = image_data.astype('float32') / 255.0
            image_data = (image_data - np.array([0.40789655, 0.44719303, 0.47026116])) / np.array([0.2886383, 0.27408165, 0.27809834])
            image_data = np.transpose(image_data, (2, 0, 1))[None]
            
            # Run inference
            with torch.no_grad():
                input_tensor = torch.from_numpy(image_data).float().to(device)
                hm, wh, offset, iou = model(input_tensor)
                
                # Decode predictions
                try:
                    outputs = decode_bbox(hm, wh, offset, iou, confidence=confidence_threshold)
                    
                    # Check if outputs is empty
                    if not outputs or len(outputs[0]) == 0:
                        print(f"No detections for image {image_id}")
                        continue
                        
                    results_boxes = postprocess(outputs, True, image_shape, input_shape, False, nms_threshold) 
                    
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
                        if class_id not in cocoGt.cats:
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
        # Load results into COCO API
        cocoDt = cocoGt.loadRes('detection_results.json')
        
        # Make sure image IDs match between GT and detections
        gt_img_ids = set(cocoGt.getImgIds())
        dt_img_ids = set(cocoDt.getImgIds())
        common_img_ids = gt_img_ids.intersection(dt_img_ids)
        
        print(f"GT has {len(gt_img_ids)} images, DT has {len(dt_img_ids)} images")
        print(f"Common images: {len(common_img_ids)}")
        
        if len(common_img_ids) == 0:
            print("No common images between ground truth and detections!")
            return 0.0
            
        # Run evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        
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

def load_saved_model(model_path, classes,device):
    """Load the model from the given path"""
    try:
        # You'll need to implement this based on your model architecture
        from mbv4_timm import CenterNet
        model = CenterNet(nc=len(classes))
        model = load_model(model,model_path)
        model.to(device)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='COCO Evaluation Script')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to validation data folder')
    parser.add_argument('--coco_gt_path', type=str, default = 'val_output_coco.json', help='Path to COCO ground truth annotations')
    parser.add_argument('--classes_file', type=str, default="classes.txt", help='Path to validation data folder')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[512, 512], help='Input shape (width, height)')
    parser.add_argument('--confidence', type=float, default=0.05, help='Confidence threshold')
    parser.add_argument('--nms', type=float, default=0.2, help='NMS threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on (cuda/cpu)')
    
    args = parser.parse_args()
    
    f = open(args.classes_file,"r").readlines()
    classes = []
    for i in f:
        classes.append(i.strip('\n'))
    # Load model
    device = torch.device(args.device)
    model = load_saved_model(args.model_path,classes, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Run evaluation
    mean_ap = evaluate_coco(
        model=model,
        device=device,
        val_data_path=args.val_data_path,
        coco_gt_path=args.coco_gt_path,
        input_shape=tuple(args.input_shape),
        confidence_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    print(f"Final mAP: {mean_ap:.4f}")

if __name__ == "__main__":
    main()
