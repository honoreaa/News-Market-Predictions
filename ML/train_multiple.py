"""
Train model multiple times with different seeds to get average performance.
This helps estimate the true accuracy and robustness of the model.
"""
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from train import main as train_main, set_seed
import json

def train_with_seed(seed, config_path, output_dir):
    """Train model with a specific seed and return results."""
    print(f"\n{'='*60}")
    print(f"Training with seed: {seed}")
    print(f"{'='*60}")
    
    #load config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    #override seed + save temporary config
    config['training']['seed'] = seed
    set_seed(seed)
    temp_config = Path('temp_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    try:
        #import and run my training script but if we want to change it
        #we'll need to modify train.py's metrics
        import subprocess
        result = subprocess.run(
            ['python3', '-m', 'ML.train', '--config', str(temp_config)],
            capture_output=True,
            text=True
        )
        
        #results
        output = result.stdout
        best_val_acc = None
        best_val_loss = None
        final_val_acc = None
        
        for line in output.split('\n'):
            if 'Val Acc:' in line:
                try:
                    acc = float(line.split('Val Acc:')[1].split(',')[0].strip())
                    if best_val_acc is None or acc > best_val_acc:
                        best_val_acc = acc
                    final_val_acc = acc
                except:
                    pass
            if 'Val Loss:' in line:
                try:
                    loss = float(line.split('Val Loss:')[1].split(',')[0].strip())
                    if best_val_loss is None or loss < best_val_loss:
                        best_val_loss = loss
                except:
                    pass
        
        return {
            'seed': seed,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'final_val_acc': final_val_acc,
            'output': output
        }
    finally:
        if temp_config.exists():
            temp_config.unlink()

def main():
    parser = argparse.ArgumentParser(description='Train model multiple times with different seeds')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of training runs')
    parser.add_argument('--seeds', type=int, nargs='+', default=None, help='Specific seeds to use')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    #generate seeds
    if args.seeds:
        seeds = args.seeds
    else:
        #use seeds 41, 67, 69, 420, 6769 lol
        seeds = [41, 67, 69, 420, 6769][:args.num_runs]
    
    print(f"Training model {len(seeds)} times with seeds: {seeds}")
    print(f"Results will be saved to: {output_dir}")
    
    results = []
    for seed in seeds:
        result = train_with_seed(seed, args.config, output_dir)
        results.append(result)
        #save individual result
        with open(output_dir / f'result_seed_{seed}.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    #calculate statistics for accuracy and loss in validation set
    val_accs = [r['best_val_acc'] for r in results if r['best_val_acc'] is not None]
    val_losses = [r['best_val_loss'] for r in results if r['best_val_loss'] is not None]
    
    stats = {
        'num_runs': len(results),
        'seeds': seeds,
        'validation_accuracy': {
            'mean': float(np.mean(val_accs)),
            'std': float(np.std(val_accs)),
            'min': float(np.min(val_accs)),
            'max': float(np.max(val_accs)),
            'values': val_accs
        },
        'validation_loss': {
            'mean': float(np.mean(val_losses)),
            'std': float(np.std(val_losses)),
            'min': float(np.min(val_losses)),
            'max': float(np.max(val_losses)),
            'values': val_losses
        }
    }
    
    # save then print summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
        print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Number of runs: {len(results)}")
    print(f"\nValidation Accuracy:")
    print(f"  Mean: {stats['validation_accuracy']['mean']:.4f}")
    print(f"  Std:  {stats['validation_accuracy']['std']:.4f}")
    print(f"  Min:  {stats['validation_accuracy']['min']:.4f}")
    print(f"  Max:  {stats['validation_accuracy']['max']:.4f}")
    print(f"\nValidation Loss:")
    print(f"  Mean: {stats['validation_loss']['mean']:.4f}")
    print(f"  Std:  {stats['validation_loss']['std']:.4f}")
    print(f"  Min:  {stats['validation_loss']['min']:.4f}")
    print(f"  Max:  {stats['validation_loss']['max']:.4f}")
    print(f"\nResults saved to: {output_dir}/")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

