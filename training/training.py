import copy
import datetime
from tqdm import tqdm
import torch


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


def show_summary_epoch(num_epochs, epoch, metrics):
    print(f"Epoch {epoch + 1}/{num_epochs} "
          f"- Train Loss: {metrics['train_losses'][-1]:.4f} "
          f"- Val Loss: {metrics['val_losses'][-1]:.4f} "
          f"- Dice Score: {metrics['dice_scores'][-1]:.4f}  "
          f"- Io Score: {metrics['iou_scores'][-1]:.4f}")


def train(model, criterion, optimizer, train_loader, num_epochs, epoch, metrics, device):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc="Training " + str(epoch + 1) + "/" + str(num_epochs)):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    metrics['train_losses'].append(running_loss / len(train_loader))
    return metrics


def validation(model, criterion, val_loader, num_epochs, epoch, metrics, device):
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    iou_score = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating" + str(epoch + 1) + "/" + str(num_epochs)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            pred = torch.sigmoid(outputs)
            dice_score += calculate_dice_score(pred, masks)
            iou_score += calculate_iou_score(pred, masks)
        metrics['val_losses'].append(val_loss / len(val_loader))
        metrics['dice_scores'].append(dice_score / len(val_loader))
        metrics['iou_scores'].append(iou_score / len(val_loader))
    return metrics


def better_validation_loss(metrics, best_val):
    return True if metrics[-1] < best_val else False


def save_model(model, best_model_wts, model_save_path):
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)


def train_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_save_path, device):
    start_time = datetime.datetime.now()
    print(f'Starting training {start_time}')
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'dice_scores': [],
        'iou_scores': []
    }
    best_val = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    model.to(device)
    for epoch in range(num_epochs):
        # Training phase
        metrics = train(model, criterion, optimizer, train_loader, num_epochs, epoch, metrics, device)
        # Validation phase
        metrics = validation(model, criterion, val_loader, num_epochs, epoch, metrics, device)
        show_summary_epoch(num_epochs, epoch, metrics)
        if better_validation_loss(metrics['val_losses'], best_val):
            best_val = metrics['val_losses'][-1]
            best_model_wts = copy.deepcopy(model.state_dict())
    save_model(model, best_model_wts, model_save_path)
    print(f'Total running time {datetime.datetime.now() - start_time}')
    return metrics
