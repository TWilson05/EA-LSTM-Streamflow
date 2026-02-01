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

def _run_epoch(model, loader, device, optimizer=None):
    """Shared core logic for training and evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_w_loss = 0
    total_valid = 0
    criterion = BasinAveragedNSELoss()
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    
    with context:
        pbar = tqdm(loader, desc="Train" if is_train else "Eval")
        for x_dyn, x_stat, y, q_std in pbar:
            x_dyn, x_stat, y, q_std = x_dyn.to(device), x_stat.to(device), y.to(device), q_std.to(device)
            
            mask = ~torch.isnan(y)
            num_valid = mask.sum().item()
            if num_valid == 0: continue
            
            if is_train: optimizer.zero_grad()
            
            y_pred = model(x_dyn, x_stat)
            loss = criterion(y_pred, y, q_std)
            
            if is_train:
                loss.backward()
                optimizer.step()
            
            # Weighted Accumulation
            total_w_loss += loss.item() * num_valid
            total_valid += num_valid
            
            # Optional: Update progress bar with current batch loss
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
    return total_w_loss / total_valid if total_valid > 0 else 0.0

def train_epoch(model, loader, optimizer, device):
    return _run_epoch(model, loader, device, optimizer)

def evaluate(model, loader, device):
    return _run_epoch(model, loader, device, optimizer=None)
