import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Modify these paths as needed
BASE_DIR = "coco"
ANNOTATION_FILE = os.path.join(BASE_DIR, "annotations_subset.json")  # Suppose you have a reduced COCO-style JSON
IMAGE_DIR = os.path.join(BASE_DIR, "images_subset")                  # Suppose images are copied here

# Image size used in the paper
IMAGE_SIZE = (224, 224)

class COCOCustomDataset(Dataset):
    """
    Custom dataset for bounding box regression as described in the paper.
    Each sample returns an image (tensor) and its bounding box coordinates [x, y, w, h].
    Some bounding boxes may be missing for semi-supervised setting.
    """
    def __init__(self, 
                 annotation_file=ANNOTATION_FILE, 
                 image_dir=IMAGE_DIR, 
                 transform=None):
        """
        Args:
            annotation_file (str): Path to a COCO-style JSON with subset or partial bounding boxes.
            image_dir (str): Directory containing images.
            transform (callable, optional): Transformations to apply to the images.
        """
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform

        # Load annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        # Build index of images and bounding boxes
        # Expecting { "images": [...], "annotations": [...] } in a COCO-like format
        self.image_id_to_file = {}
        for img_info in coco_data["images"]:
            self.image_id_to_file[img_info["id"]] = img_info["file_name"]

        self.data = []
        for ann in coco_data["annotations"]:
            image_id = ann["image_id"]
            # If bounding box is partially available
            if "bbox" in ann:
                bbox = ann["bbox"]  # [x, y, w, h]
                self.data.append({
                    "image_id": image_id,
                    "bbox": bbox
                })
            else:
                # If missing due to semi-supervised drop-out
                self.data.append({
                    "image_id": image_id,
                    "bbox": None
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_id = sample["image_id"]
        bbox = sample["bbox"]

        # Load image
        img_file = self.image_id_to_file[image_id]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert bounding box to tensor if it exists, else return None
        if bbox is not None:
            bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        else:
            bbox_tensor = None

        return image, bbox_tensor


def get_transforms():
    """
    Returns the composition of transformations used in the paper:
    1. Resize to 224x224
    2. ToTensor
    3. Normalize with ImageNet means/stdev
    """
    return T.Compose([
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]),
    ])
