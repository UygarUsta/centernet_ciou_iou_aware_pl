# Custom CenterNet Model Reimplementation

This repository contains a reimplementation of a custom CenterNet model with various enhancements and features for object detection tasks.

## Features

- **Model Backbone**: MobileNetV4 with Feature Pyramid Network (FPN).
- **Loss Function**: CIOU (Complete Intersection over Union) Loss.
- **IOU Aware Head**: IOU aware head for better predictions.
- **Augmentation Techniques**:
  - Mosaic Augmentation
  - MixUp Augmentation


---

## Data Directory Structure

To train the model, organize your dataset in the following structure:

```
DataRoot/
├── train_images/
│   ├── 1.jpg
│   ├── 1.xml
├── val_images/
│   ├── 1.jpg
│   ├── 1.xml
```

- Each image should have a corresponding annotation file in `.xml` format (e.g., Pascal VOC format).

---

## Important Notes

1. **Dataset Preparation**:
   - If you plan to train the model with a new dataset, ensure that you delete the following files if they are created:
     - `classes.txt`
     - Any `.json` files in the data directory.
   - These files are automatically generated and specific to the current dataset.

2. **Augmentations**:
   - Mosaic and MixUp augmentations are applied during training for better generalization and robustness.

3. **Model Configuration**:
   - The model uses MobileNetV4 as the backbone, combined with an FPN for multi-scale feature extraction.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/UygarUsta/centernet_ciou_iou_aware_pl.git
   cd centernet_ciou_iou_aware_pl
   ```

2. Prepare your dataset following the structure mentioned above.

3. Delete `classes.txt` and `.json` files if training on a new dataset.

4. Run the training script:
   ```bash
   python train.py --data_dir /path/to/DataRoot
   ```

---

## Acknowledgments

This project is inspired by the original CenterNet paper and aims to provide a clean and modular implementation with additional features for better performance.

- https://github.com/xingyizhou/CenterNet
- https://github.com/610265158/mobile_centernet
- https://github.com/bubbliiiing/centernet-pytorch
---

## License

[MIT License](LICENSE)

Feel free to modify and use this repository for your projects!
