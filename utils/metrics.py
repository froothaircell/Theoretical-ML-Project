import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def smooth_l1_loss(preds, targets):
    """
    Smooth L1 Loss used for bounding box regression.
    pred, target shapes: (B, 4)
    """
    return F.smooth_l1_loss(preds, targets, reduction='mean')

def collate_fn(batch):
    """
    Collate function for DataLoader when bounding boxes may be None (semi-supervised).
    Returns:
        images: list of image tensors
        targets: list of bounding box tensors (some may be None)
    """
    images = []
    bboxes = []
    for img, bbox in batch:
        images.append(img)
        bboxes.append(bbox)

    images = torch.stack(images, dim=0)

    # We keep bounding boxes as is (list) because some are None
    return images, bboxes

def compute_metrics(pred_bboxes, true_bboxes):
    """
    Compute MSE, MAE, and R^2 for bounding box regression.
    pred_bboxes, true_bboxes shape: (N, 4) each
    Return: dict with "mse", "mae", "r2"
    """
    # pred_bboxes, true_bboxes => numpy
    pred = pred_bboxes.cpu().numpy()
    true = true_bboxes.cpu().numpy()

    mse = np.mean((pred - true)**2)
    mae = np.mean(np.abs(pred - true))

    # R^2
    var = np.mean((true - np.mean(true, axis=0))**2)
    r2 = 1 - (mse / var if var > 0 else 1)

    return {"mse": mse, "mae": mae, "r2": r2}
