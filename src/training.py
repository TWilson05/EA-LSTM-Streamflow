import torch
import numpy as np
from tqdm import tqdm

class NSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Mask: Create a boolean mask where y_true is NOT nan
        mask = ~torch.isnan(y_true)
        
        # Apply mask
        pred = y_pred[mask]
        true = y_true[mask]
        
        if len(true) == 0:
            return torch.tensor(0.0, requires_grad=True).to(y_pred.device)
        
        # NSE Calculation
        # Numerator: Sum of Squared Errors
        numerator = torch.sum((true - pred) ** 2)
        
        # Denominator: Sum of Squared differences from mean
        true_mean = torch.mean(true)
        denominator = torch.sum((true - true_mean) ** 2)
        
        # Add epsilon to avoid div by zero
        nse = 1 - (numerator / (denominator + 1e-6))
        
        # Loss is 1 - NSE (so we minimize loss to maximize NSE)
        # Note: If NSE is negative (model worse than mean), loss > 1. 
        # Range of NSE is (-inf, 1]. Range of Loss is [0, inf).
        return numerator / (denominator + 1e-6)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = NSELoss()
    
    for x_dyn, x_stat, y in tqdm(loader, desc="Training"):
        x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_pred = model(x_dyn, x_stat)
        
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    total_nse = 0
    count = 0
    
    with torch.no_grad():
        for x_dyn, x_stat, y in loader:
            x_dyn, x_stat, y = x_dyn.to(device), x_stat.to(device), y.to(device)
            y_pred = model(x_dyn, x_stat)
            
            # Helper to calculate actual NSE for reporting
            mask = ~torch.isnan(y)
            if mask.sum() == 0: continue
            
            p = y_pred[mask]
            t = y[mask]
            
            num = torch.sum((t - p)**2)
            den = torch.sum((t - t.mean())**2)
            nse = 1 - (num / (den + 1e-6))
            
            total_nse += nse.item()
            count += 1
            
    return total_nse / count if count > 0 else 0
