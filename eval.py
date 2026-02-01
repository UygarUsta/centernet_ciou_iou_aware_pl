import os
import glob
import json
import cv2
import numpy as np
import torch
import faster_coco_eval
# This single line replaces pycocotools with faster-coco-eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
from infer_utils import load_model
from utils_bbox import decode_bbox, postprocess
from tqdm import tqdm 
from lightning_model import COCOEvalDataset
from torch.utils.data import DataLoader,Dataset
from PIL import Image 

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
    


def evaluate_coco(model, device, val_data_path, coco_gt_path,classes,input_shape=(512, 512), confidence_threshold=0.05, nms_threshold=0.2):
        """High-performance COCO evaluation using Batched DataLoader"""
        
        cocoGt = COCO(coco_gt_path)
        if not cocoGt:
            print("Failed to load COCO ground truth")
            return 0.0

        # Mapping: Class Name -> COCO ID
        cat_name_to_id = {cat['name']: cat['id'] for cat in cocoGt.cats.values()}
        
        val_images_folder = val_data_path
        
        # 1. Create the optimized DataLoader
        eval_dataset = COCOEvalDataset2(cocoGt, val_images_folder, input_shape)
        
        # Use a reasonable batch size (e.g., 32) and workers (e.g., 4 or 8)
        eval_loader = DataLoader(
            eval_dataset, 
            batch_size=16,       # Increase this if you have GPU memory
            shuffle=False, 
            num_workers=4,       # Parallel loading
            pin_memory=True
        )

        print(f"Starting evaluation on {len(eval_dataset)} images...")
        
        model.eval()
        results = []
        
        # 2. Batched Inference Loop
        with torch.no_grad():
            for batch_imgs, batch_ids, batch_orig_h, batch_orig_w in tqdm(eval_loader, desc="Evaluating"):
                # Move batch to GPU
                batch_imgs = batch_imgs.to(device)
                
                # Inference
                hm, wh, offset, iou = model(batch_imgs)
                
                # Decode bounding boxes
                # Note: decode_bbox processes the whole batch at once
                outputs = decode_bbox(hm, wh, offset, iou, confidence=confidence_threshold)
                
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
                    results_boxes = postprocess([output], True, image_shape, input_shape, True, nms_threshold)
                    
                    # Format for JSON
                    for box in results_boxes[0]:
                        if len(box) < 6: continue
                        y1, x1, y2, x2, conf, cls_id = box
                        
                        
                        if x2 <= x1 or y2 <= y1: continue
                        
                        # ID Mapping Logic
                        pred_class_idx = int(cls_id)
                        if pred_class_idx >= len(classes): continue
                        
                        pred_class_name = classes[pred_class_idx]
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
        with open('detection_results_eval.json', 'w') as f:
            json.dump(results, f)
            
        try:
            cocoDt = cocoGt.loadRes('detection_results_eval.json')
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            return cocoEval
        except Exception as e:
            print(f"COCO evaluation error: {e}")
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
    

def load_saved_model_hardnet(model_path, classes, device):
    """Load the model from the given path"""
    from hardnet import get_pose_net
    model = get_pose_net(68,{"hm":len(classes),"wh":2,"offset":2,"iou":1}) 
    if model_path.endswith(".ckpt"):
        #model = hardnet_load_model(model,model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "") 
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model 


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
    #model = load_saved_model(args.model_path,classes, device)
    model = load_saved_model_hardnet(args.model_path,classes,device)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Run evaluation
    mean_ap = evaluate_coco(
        model=model,
        device=device,
        val_data_path=args.val_data_path,
        coco_gt_path=args.coco_gt_path,
        classes=classes,
        input_shape=tuple(args.input_shape),
        confidence_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    

if __name__ == "__main__":
    main()
