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
    
        
def eval_model(model, data_loader):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
    MODEL_PATH = './models/news_classifier.pt'
    
    model.eval()
    

    results_true = []
    results_prob = []

    with torch.no_grad():
        for batch in data_loader:
            emb = batch["emb"].cpu()
            price = batch["price"].cpu()
            labels = batch["label"].cpu()

            probs = model(emb, price)

            results_true.append(labels.cpu())
            results_prob.append(probs.cpu())

    y_true = torch.cat(results_true).numpy()
    y_prob = torch.cat(results_prob).numpy()
    y_pred = (y_prob > 0.5).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "classification_report": classification_report(y_true, y_pred),
    }, y_true, y_prob

def plot_evaluations(evaluations, y_true, y_prob):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    
    # auc
    false_pos_rate, true_pos_rate, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(false_pos_rate, true_pos_rate)
    plt.figure()
    plt.plot(false_pos_rate, true_pos_rate, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC")
    plt.legend()
    plt.grid()
    plt.show()
    
    # precision recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve Plot")
    plt.grid()
    plt.show()
    
    # prob distirbutions
    plt.figure()
    plt.hist(y_prob, bins=25)
    plt.title("Probabilities")
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

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
            
            #similar to what we did in class on the slides
            #change the predictions threshold to make stricter or less stricter
            train_loss += loss.item()
            predictions = (outputs > 0.50).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        #validation phase
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
    
    device_str = config['training']['device']
    device_str = 'cpu'
    device = torch.device(device_str)
    
    embeddings_dir = Path(config['data']['embeddings_dir'])
    embeddings_npz = embeddings_dir / 'embeddings.npz'
    meta_csv = embeddings_dir / 'meta.csv'
    
    if not embeddings_npz.exists() or not meta_csv.exists():
        raise FileNotFoundError(f"Embeddings not found. You didn't run embedding.py first :( go do that!!! :) ")
    
    meta_df = pd.read_csv(meta_csv, parse_dates=['Date'])
    
    # split by date to avoid lookahead bias
    train_mask = meta_df['Date'] < config['training']['train_split_date']
    val_mask = (meta_df['Date'] >= config['training']['train_split_date']) & \
               (meta_df['Date'] < config['training']['val_split_date'])
    
    train_indices = meta_df[train_mask].index.tolist()
    val_indices = meta_df[val_mask].index.tolist()    
    train_dataset = NewsDataset(str(embeddings_npz), str(meta_csv), indices=train_indices)
    val_dataset = NewsDataset(str(embeddings_npz), str(meta_csv), indices=val_indices)
    
    # compute normalization stats from training set only (no lookahead bias)
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
    
 # generate evaluation metrics plots
    evals, y_true, y_prob = eval_model(model, val_loader)
    print(evals["classification_report"])
    plot_evaluations(evals, y_true, y_prob)


if __name__ == '__main__':
    main()