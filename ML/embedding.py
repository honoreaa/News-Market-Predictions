import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
from tqdm import tqdm

def get_model_and_tokenizer(model_name):
    # load the pretrained model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def embed_texts(texts, tokenizer, model, device='cpu', batch_size=16, max_length=128):
    # auto detect device if needed
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    model.to(device)
    model.eval()
    
    # try compiling for speed if pytorch 2.0+
    try:
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
    except Exception:
        pass
    
    embeddings = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():  # dont need gradients for inference
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", total=num_batches):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            for k in encoded:
                encoded[k] = encoded[k].to(device)
            out = model(**encoded)
            # mean pooling over tokens to get single vector per text
            last_hidden = out.last_hidden_state
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            summed = (last_hidden * attention_mask).sum(1)
            counts = attention_mask.sum(1)
            pooled = summed / counts.clamp(min=1)  # avoid div by zero
            pooled = pooled.cpu().numpy()
            embeddings.append(pooled)
    embeddings = np.vstack(embeddings)
    return embeddings

# finbert works well for financial news
def main(in_csv, out_dir, model_name='yiyanghkust/finbert-tone', batch_size=16, max_length=128, device='auto'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv, parse_dates=['Date'])
    texts = df['headlines'].fillna("").astype(str).tolist()

    tokenizer, model = get_model_and_tokenizer(model_name)
    embs = embed_texts(texts, tokenizer, model, device=device, batch_size=batch_size, max_length=max_length)

    np.savez_compressed(out_dir / 'embeddings.npz', embeddings=embs)
    price_cols = ['Date','ticker','label','Close','close_return_1d','vol_log',
                  'close_return_3d','close_return_7d','volatility_7d','price_position','volume_trend']
    available_cols = [col for col in price_cols if col in df.columns]
    df_out = df[available_cols].copy()
    df_out.to_csv(out_dir / 'meta.csv', index=False)
    print('Saved embeddings and meta')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_csv', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='data/embeddings')
    parser.add_argument('--model_name', type=str, default='yiyanghkust/finbert-tone')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto', help='Device to use: auto, cpu, cuda, or mps (auto detects best available)')
    args = parser.parse_args()
    main(args.in_csv, args.out_dir, model_name=args.model_name, batch_size=args.batch_size, max_length=args.max_length, device=args.device)