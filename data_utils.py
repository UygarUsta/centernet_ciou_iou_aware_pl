from collections import defaultdict
from tqdm import tqdm 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json 
import xml.etree.ElementTree as ET
import os 

def xml_to_coco_json(xml_dir, output_json_path):
    coco_data = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    category_dict = {}
    annotation_id = 1

    # Process each XML file
    for xml_file in tqdm(os.listdir(xml_dir)):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            
            # Gather image data
            filename = root.find('filename').text #+ ".jpg" #'.jpg' is temporary, due to lack of '.jpg' in xml files of COCO
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            image_id = len(coco_data["images"]) + 1
            coco_data["images"].append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename
            })
            
            # Process each object in the image
            for obj in root.findall('object'):
                category = obj.find('name').text
                if category not in category_dict:
                    category_dict[category] = len(category_dict) + 1
                    coco_data["categories"].append({
                        "id": category_dict[category],
                        "name": category
                    })
                
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                coco_data["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_dict[category],
                    "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save to JSON
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_data, json_file, indent=4)

    print(f"Converted annotations saved to {output_json_path}")


def extract_coordinates(file_path,classes):
    tree = ET.parse(file_path)
    root = tree.getroot()
    all_objects_coords = []
    size = root.find('size')
    image_width = int(size.find('width').text) 
    image_height = int(size.find('height').text)
    for obj in root.findall('object'):
        coords = []    
        # Extracting coordinates from bndbox
        bndbox = obj.find('bndbox')
        name = obj.find('name').text
        if bndbox is not None:
            xmin = int(float(str(bndbox.find('xmin').text)))
            ymin = int(float(str(bndbox.find('ymin').text)))
            xmax = int(float(str(bndbox.find('xmax').text)))
            ymax = int(float(str(bndbox.find('ymax').text)))
            coords.append(xmin)
            coords.append(ymin)
            coords.append(xmax)
            coords.append(ymax)
            coords.append(classes.index(name))


        all_objects_coords.append(coords)


    return all_objects_coords


def visualize_data_loading(data_module, classes, num_samples=5): 
    """
    Used in train_lightning.py for visualizing augmented data.
    """
    import cv2
    import numpy as np
    print("Visualizing data loading with augmentations...")
    
    # Prepare data module
    data_module.prepare_data()
    data_module.setup("fit")
    
    # Get training dataloader
    train_dataloader = data_module.train_dataloader()
    
    # Define colors for visualization
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), 
              (0, 255, 255), (255, 0, 255), (128, 0, 0), (0, 128, 0)]
    
    sample_count = 0
    for batch in train_dataloader:
        images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch
        
        for i in range(min(images.shape[0], 3)):  # Show up to 3 images from each batch
            # Get image and convert to numpy array for visualization
            image = images[i].permute(1, 2, 0).cpu().numpy()
            # Denormalize image
            image = (image * 255) #.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = image + [0.40789655, 0.44719303, 0.47026116]
            image = image * [0.2886383, 0.27408165, 0.27809834]
            
            # Convert tensors to numpy arrays
            hms = batch_hms[i].cpu().numpy()
            whs = batch_whs[i].cpu().numpy()
            regs = batch_regs[i].cpu().numpy()
            reg_masks = batch_reg_masks[i].cpu().numpy()
            
            # Find all objects in the image
            for c in range(hms.shape[-1]):
                heatmap = hms[..., c]
                
                # Find peaks in the heatmap where reg_mask > 0
                indices = np.where((heatmap > 0.5) & (reg_masks > 0))
                
                for y, x in zip(indices[0], indices[1]):
                    # Get width and height
                    w, h = whs[y, x]
                    # Get offset
                    offset_x, offset_y = regs[y, x]
                    
                    # Calculate center point with offset
                    cx = int((x + offset_x))
                    cy = int((y + offset_y))
                    
                    # Calculate bounding box coordinates
                    x1 = int(cx - w/2) * data_module.stride
                    y1 = int(cy - h/2) * data_module.stride
                    x2 = int(cx + w/2) * data_module.stride
                    y2 = int(cy + h/2) * data_module.stride
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(image.shape[1] - 1, x2)
                    y2 = min(image.shape[0] - 1, y2)
                    # Draw bounding box
                    color = colors[c % len(colors)]
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add class label
                    class_name = classes[c]
                    cv2.putText(image, class_name, (x1, y1 - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display image
            cv2.imshow(f"Sample {sample_count + 1}", image)
            cv2.waitKey(1)  # Update display
            sample_count += 1
            
            # Break if we've shown enough samples
            if sample_count >= num_samples:
                break
        
        if sample_count >= num_samples:
            break
    
    print("Press 'q' to continue to training or close all windows...")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
            
        # Try to check if windows are still open
        try:
            if cv2.getWindowProperty(f"Sample 1", cv2.WND_PROP_VISIBLE) < 1:
                break
        except:
            # Window might be closed already
            break
    
    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("Starting training...")