import torch
from tqdm import tqdm

class BasinAveragedNSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, q_std):
        """
        y_pred: (Batch, 1)
        y_true: (Batch, 1)
        q_std:  (Batch, 1) - Pre-computed observation std for each basin in batch
        """
        # Mask NaNs
        mask = ~torch.isnan(y_true)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True).to(y_pred.device)
            
        pred = y_pred[mask]
        true = y_true[mask]
        std = q_std[mask]
        
        # Basin-Averaged NSE* Loss:
        squared_error = (pred - true) ** 2
        variance = (std + 1e-6) ** 2
        
        normalized_squared_error = squared_error / variance
        
        # Return the mean over the batch
        return torch.mean(normalized_squared_error)

def train_epoch(model, loader, optimizer, device):
    model.train()
    
    total_accumulated_loss = 0.0  # Sum of (Batch_Loss * Num_Valid_Samples)
    total_valid_samples = 0       # Sum of Num_Valid_Samples
    
    criterion = BasinAveragedNSELoss()
    
    for x_dyn, x_stat, y, q_std in tqdm(loader, desc="Training"):
        x_dyn, x_stat, y, q_std = x_dyn.to(device), x_stat.to(device), y.to(device), q_std.to(device)
        
        # 1. Calculate Mask & Count Valid Samples
        # We need this count to weigh the batch correctly
        mask = ~torch.isnan(y)
        num_valid = mask.sum().item()
        
        # Skip batch if no valid data to avoid messing up gradients
        if num_valid == 0:
            continue
            
        optimizer.zero_grad()
        y_pred = model(x_dyn, x_stat)
        
        # 2. Compute Loss (This returns the Mean of the batch)
        loss = criterion(y_pred, y, q_std)
        loss.backward()
        optimizer.step()
        
        # 3. Accumulate Weighted Loss
        # We multiply by num_valid to "undo" the mean inside the loss function
        # giving us the total Sum of Errors for this batch
        total_accumulated_loss += loss.item() * num_valid
        total_valid_samples += num_valid
        
    # 4. Compute Global Average
    if total_valid_samples > 0:
        return total_accumulated_loss / total_valid_samples
    else:
        return 0.0

def evaluate(model, loader, device):
    """
    Calculates the Basin-Averaged NSE Loss on the validation/test set.
    Returns: Weighted Average Loss (Lower is Better)
    """
    model.eval()
    total_accumulated_loss = 0.0
    total_valid_samples = 0
    
    criterion = BasinAveragedNSELoss()
    
    with torch.no_grad():
        # We need q_std here for the loss calculation
        for x_dyn, x_stat, y, q_std in loader:
            x_dyn, x_stat, y, q_std = x_dyn.to(device), x_stat.to(device), y.to(device), q_std.to(device)
            
            # 1. Mask & Count
            mask = ~torch.isnan(y)
            num_valid = mask.sum().item()
            
            if num_valid == 0: continue
            
            # 2. Predict
            y_pred = model(x_dyn, x_stat)
            
            # 3. Compute Loss
            loss = criterion(y_pred, y, q_std)
            
            # 4. Accumulate Weighted
            total_accumulated_loss += loss.item() * num_valid
            total_valid_samples += num_valid
            
    if total_valid_samples > 0:
        return total_accumulated_loss / total_valid_samples
    else:
        return 0.0
