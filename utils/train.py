import torch
from torch.utils.data import DataLoader
from training.train_utils import smooth_l1_loss, collate_fn, compute_metrics

def train_one_epoch(model, 
                    optimizer, 
                    data_loader, 
                    device="cpu"):
    model.train()
    total_loss = 0.0
    count = 0
    for images, bboxes in data_loader:
        # Move data to device
        images = images.to(device)

        # Filter out samples with bounding boxes = None (unlabeled) in semi-supervised
        # We only backprop on labeled data
        labeled_indices = [i for i, bb in enumerate(bboxes) if bb is not None]
        if len(labeled_indices) == 0:
            # If no labeled examples in this batch, skip
            continue

        labeled_images = images[labeled_indices]
        labeled_bboxes = [bboxes[i] for i in labeled_indices]
        labeled_bboxes = torch.stack(labeled_bboxes, dim=0).to(device)

        # Forward pass
        preds = model(labeled_images)  # shape: (B_labeled, 4)
        loss = smooth_l1_loss(preds, labeled_bboxes)

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labeled_indices)
        count += len(labeled_indices)

    if count == 0:
        return 0.0
    return total_loss / count

@torch.no_grad()
def validate_one_epoch(model, 
                       data_loader, 
                       device="cpu"):
    model.eval()
    total_loss = 0.0
    count = 0
    all_preds = []
    all_trues = []

    for images, bboxes in data_loader:
        images = images.to(device)

        # Forward pass for all images (even unlabeled)
        preds = model(images)  # shape: (B, 4)
        
        # For those that are labeled, compute loss & metrics
        labeled_indices = [i for i, bb in enumerate(bboxes) if bb is not None]
        if len(labeled_indices) > 0:
            labeled_preds = preds[labeled_indices]
            labeled_bboxes = [bboxes[i] for i in labeled_indices]
            labeled_bboxes = torch.stack(labeled_bboxes, dim=0).to(device)

            loss = smooth_l1_loss(labeled_preds, labeled_bboxes)
            total_loss += loss.item() * len(labeled_indices)
            count += len(labeled_indices)

            all_preds.append(labeled_preds)
            all_trues.append(labeled_bboxes)

    if count == 0:
        return 0.0, {"mse": 0.0, "mae": 0.0, "r2": 0.0}

    # Compute average loss
    avg_loss = total_loss / count

    # Compute MSE, MAE, R^2
    all_preds = torch.cat(all_preds, dim=0)
    all_trues = torch.cat(all_trues, dim=0)
    metrics = compute_metrics(all_preds, all_trues)

    return avg_loss, metrics
