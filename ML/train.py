import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import yaml
from pathlib import Path

try:
    from .dataset import NewsDataset
    from .model import NewsClassifier
except ImportError:
    from dataset import NewsDataset
    from model import NewsClassifier

def set_seed(seed=42):
    # set all the random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, device, epochs, lr):
    # simple gradient descent with adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            emb = batch['emb'].to(device)
            price = batch['price'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(emb, price)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                emb = batch['emb'].to(device)
                price = batch['price'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(emb, price)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['training']['seed'])
    
    device_str = config['training']['device']
    if device_str == 'auto':
        if torch.cuda.is_available():
            device_str = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = 'mps'
        else:
            device_str = 'cpu'
    
    device = torch.device(device_str)
    print(f'Using device: {device}')
    
    embeddings_dir = Path(config['data']['embeddings_dir'])
    embeddings_npz = embeddings_dir / 'embeddings.npz'
    meta_csv = embeddings_dir / 'meta.csv'
    
    if not embeddings_npz.exists() or not meta_csv.exists():
        raise FileNotFoundError(f"Embeddings not found. Please run embedding.py first.")
    
    meta_df = pd.read_csv(meta_csv, parse_dates=['Date'])
    
    # split by date to avoid lookahead bias
    train_mask = meta_df['Date'] < config['training']['train_split_date']
    val_mask = (meta_df['Date'] >= config['training']['train_split_date']) & \
               (meta_df['Date'] < config['training']['val_split_date'])
    
    train_indices = meta_df[train_mask].index.tolist()
    val_indices = meta_df[val_mask].index.tolist()
    
    print(f'Train samples: {len(train_indices)}, Val samples: {len(val_indices)}')
    
    train_dataset = NewsDataset(str(embeddings_npz), str(meta_csv), indices=train_indices)
    val_dataset = NewsDataset(str(embeddings_npz), str(meta_csv), indices=val_indices)
    
    # compute normalization stats from training set only (no lookahead bias)
    print('Computing feature normalization statistics from training set...')
    train_price_features = []
    # sample a subset for efficiency
    for i in range(min(1000, len(train_dataset))):
        sample = train_dataset[i]
        train_price_features.append(sample['price'].numpy())
    train_price_features = np.array(train_price_features)
    
    price_mean = torch.tensor(train_price_features.mean(axis=0), dtype=torch.float32)
    price_std = torch.tensor(train_price_features.std(axis=0) + 1e-8, dtype=torch.float32)
    
    class NormalizedDataset:
        def __init__(self, dataset, mean, std):
            self.dataset = dataset
            self.mean = mean
            self.std = std
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            sample = self.dataset[idx]
            normalized_price = (sample['price'] - self.mean) / self.std
            return {
                'emb': sample['emb'],
                'price': normalized_price,
                'label': sample['label']
            }
    
    train_dataset_norm = NormalizedDataset(train_dataset, price_mean, price_std)
    val_dataset_norm = NormalizedDataset(val_dataset, price_mean, price_std)
    
    train_loader = DataLoader(train_dataset_norm, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset_norm, batch_size=config['model']['batch_size'], shuffle=False)
    
    price_dim = train_dataset_norm[0]['price'].shape[0]
    
    model = NewsClassifier(
        emb_dim=768,
        price_dim=price_dim,
        hidden_dim=config['model']['hidden_dim'],
        dropout=config['model']['dropout']
    )
    
    model = train_model(
        model, train_loader, val_loader, device,
        epochs=config['model']['epochs'],
        lr=float(config['model']['lr'])
    )
    
    model_path = Path('models')
    model_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path / 'news_classifier.pt')
    print(f'Model saved to {model_path / "news_classifier.pt"}')


if __name__ == '__main__':
    main()