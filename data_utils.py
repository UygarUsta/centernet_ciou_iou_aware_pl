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