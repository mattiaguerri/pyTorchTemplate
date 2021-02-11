import numpy as np
import torch
from torch.utils.data import Dataset


# define custom dataset
class CustomDataset(Dataset):
    
    def __init__(self, features, targets):
        self.X = features
        self.y = targets
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        x = self.X[idx, :]
        y = self.y[idx, :]
        
        x = np.float32(x)
        y = np.float32(y)
        
        return x, y