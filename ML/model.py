import torch
import torch.nn as nn


class NewsClassifier(nn.Module):
    def __init__(self, emb_dim=768, price_dim=7, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        #super chill feedforward network
        self.net = nn.Sequential(
            nn.Linear(emb_dim + price_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, emb, price):
        x = torch.cat([emb, price], dim=1)
        return self.net(x).squeeze(1)