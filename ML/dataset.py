import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class NewsDataset(Dataset):
    def __init__(self, embeddings_npz, meta_csv, indices=None):
        data = np.load(embeddings_npz)
        self.embeddings = data['embeddings']
        meta = pd.read_csv(meta_csv, parse_dates=['Date'])       
        # make sure lengths match up otherwise I get an error
        # ^ bug fixed
        min_len = min(len(self.embeddings), len(meta))
        if len(self.embeddings) != len(meta):
            self.embeddings = self.embeddings[:min_len]
            meta = meta.iloc[:min_len].reset_index(drop=True)
        
        self.indices = [i for i in indices if i < min_len] # indices to use
        self.meta = meta

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        emb = self.embeddings[i]
        row = self.meta.iloc[i]
        
        # collect all the price features we have
        price_feat = []
        for feat_name in ['close_return_1d', 'vol_log', 'close_return_3d', 'close_return_7d', 
                          'volatility_7d', 'price_position', 'volume_trend']:
            if feat_name in row.index:
                price_feat.append(float(row[feat_name]))
        
        price_feat = np.array(price_feat, dtype=float)
        label = int(row['label'])
        
        #source: https://docs.pytorch.org/docs/main/tensors.html 
        return {
            'emb': torch.tensor(emb, dtype=torch.float32),
            'price': torch.tensor(price_feat, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }