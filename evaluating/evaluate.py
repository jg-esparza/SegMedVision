import torch
from tqdm import tqdm

def calculate_dice_score(pred, targets, threshold=0.5, eps=1e-7):
    pred = (pred > threshold).float()
    targets = targets.float()
    intersection = (pred * targets).sum()
    total_sum = pred.sum() + targets.sum() + eps
    dice = (2. * intersection + eps) / total_sum
    return dice.item()


def calculate_iou_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    targets = targets.float()
    intersect = (pred * targets).sum()
    union = pred.sum() + targets.sum() - intersect
    return (intersect/union).item()


def precision_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    intersect = (pred * targets).sum()
    total_pixel_pred = pred.sum()
    precision = intersect/total_pixel_pred
    return precision.item()


def recall_score(pred, targets, threshold=0.5):
    pred = (pred > threshold).float()
    intersect = (pred*targets).sum()
    total_pixel_truth = targets.sum()
    recall = intersect/total_pixel_truth
    return recall.item()


def calculate_metrics(outputs, masks, metrics, threshold):
    pred = torch.sigmoid(outputs)
    metrics['dice_scores'].append(calculate_dice_score(pred, masks, threshold))
    metrics['iou_scores'].append(calculate_iou_score(pred, masks, threshold))
    metrics['precision_scores'].append(precision_score(pred, masks, threshold))
    metrics['recall_scores'].append(recall_score(pred, masks, threshold))
    return metrics


def calculate_average_metrics(metrics):
    return sum(metrics) / len(metrics)


def show_result(dice_scores, iou_scores, precisions, recalls):
    avg_dice = calculate_average_metrics(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    print(f"📌 Test Dice Score: {avg_dice:.4f}")
    print(f"📌 Test IoU Score: {avg_iou:.4f}")
    print(f"📌 Test Precision: {avg_precision:.4f}")
    print(f"📌 Test Recall: {avg_recall:.4f}")


def evaluate_model(model, dataloader, device, threshold=0.5):
    model.eval()
    model.to(device)
    metrics = {
        'dice_scores': [],
        'iou_scores': [],
        'precision_scores': [],
        'recall_scores': []
    }
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            metrics = calculate_metrics(outputs, masks, metrics, threshold)
        show_result(metrics['dice_scores'], metrics['iou_scores'],
                    metrics['precision_scores'], metrics['recall_scores'])
