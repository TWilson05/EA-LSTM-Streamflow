import torch
import numpy as np
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
        
        # Basin-Averaged NSE* Loss Formula:
        # Loss = Sum( (y_hat - y)^2 / (std + eps)^2 )
        # We perform element-wise division by the specific basin variance
        
        squared_error = (pred - true) ** 2
        variance = (std + 1e-6) ** 2
        
        normalized_squared_error = squared_error / variance
        
        # Return the mean over the batch
        return torch.mean(normalized_squared_error)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = BasinAveragedNSELoss()
    
    # Unpack 4 items now (including q_std)
    for x_dyn, x_stat, y, q_std in tqdm(loader, desc="Training"):
        x_dyn, x_stat, y, q_std = x_dyn.to(device), x_stat.to(device), y.to(device), q_std.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_dyn, x_stat)
        
        # Pass q_std to loss
        loss = criterion(y_pred, y, q_std)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    """
    Evaluates on Validation/Test set. 
    Returns average NSE (not loss) across the set.
    """
    model.eval()
    total_nse = 0
    count = 0
    
    with torch.no_grad():
        for x_dyn, x_stat, y, _ in loader: # Ignore q_std for reporting standard NSE
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            y_pred = model(x_dyn, x_stat)
            
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            
            p = y_pred[mask]
            t = y[mask]
            
            # Standard NSE Calculation for reporting
            num = torch.sum((t - p)**2)
            den = torch.sum((t - t.mean())**2)
            nse = 1 - (num / (den + 1e-6))
            
            total_nse += nse.item()
            count += 1
            
    return total_nse / count if count > 0 else 0
