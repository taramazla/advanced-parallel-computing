"""
Generate summary report from multiple LoRA experiments
Compares training metrics across different configurations
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment summary report")
    parser.add_argument("--experiments_dir", type=str, required=True,
                        help="Directory containing experiment outputs")
    parser.add_argument("--output_file", type=str, default="experiment_summary.txt",
                        help="Output file for summary report")
    return parser.parse_args()


def extract_metrics_from_file(metrics_file: str) -> Dict:
    """Extract metrics from training_metrics.txt file"""
    metrics = {}
    
    if not os.path.exists(metrics_file):
        return metrics
    
    with open(metrics_file, 'r') as f:
        content = f.read()
        
        # Extract configuration parameters
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['Model', 'Dataset', 'LoRA r', 'LoRA alpha', 'LoRA dropout',
                          'Epochs', 'Batch size', 'Learning rate', 'Quantization']:
                    metrics[key] = value
    
    return metrics


def extract_training_history(metrics_file: str) -> List[Dict]:
    """Extract training history from metrics file"""
    history = []
    
    if not os.path.exists(metrics_file):
        return history
    
    with open(metrics_file, 'r') as f:
        content = f.read()
        
        # Find training history section
        if 'Training History:' in content:
            history_section = content.split('Training History:')[1]
            
            # Parse each log entry
            for line in history_section.split('\n'):
                if line.strip().startswith('{'):
                    try:
                        log_entry = eval(line.strip())
                        if isinstance(log_entry, dict):
                            history.append(log_entry)
                    except:
                        pass
    
    return history


def get_final_metrics(history: List[Dict]) -> Dict:
    """Get final training and validation metrics"""
    final_metrics = {
        'final_train_loss': None,
        'final_eval_loss': None,
        'best_eval_loss': None,
        'total_steps': 0
    }
    
    if not history:
        return final_metrics
    
    # Find final losses
    train_losses = [h.get('loss') for h in history if 'loss' in h and h.get('loss') is not None]
    eval_losses = [h.get('eval_loss') for h in history if 'eval_loss' in h and h.get('eval_loss') is not None]
    
    if train_losses:
        final_metrics['final_train_loss'] = train_losses[-1]
    
    if eval_losses:
        final_metrics['final_eval_loss'] = eval_losses[-1]
        final_metrics['best_eval_loss'] = min(eval_losses)
    
    # Get total steps
    steps = [h.get('step') for h in history if 'step' in h]
    if steps:
        final_metrics['total_steps'] = max(steps)
    
    return final_metrics


def generate_report(experiments_dir: str, output_file: str):
    """Generate comprehensive experiment summary report"""
    experiments_path = Path(experiments_dir)
    
    # Find all experiment directories
    experiment_dirs = [d for d in experiments_path.iterdir() 
                      if d.is_dir() and d.name.startswith('exp')]
    
    experiment_dirs = sorted(experiment_dirs, key=lambda x: x.name)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {experiments_dir}")
        return
    
    # Collect data from all experiments
    experiments_data = []
    
    for exp_dir in experiment_dirs:
        metrics_file = exp_dir / 'training_metrics.txt'
        
        exp_data = {
            'name': exp_dir.name,
            'path': str(exp_dir),
            'config': extract_metrics_from_file(str(metrics_file)),
            'history': extract_training_history(str(metrics_file)),
        }
        
        # Get final metrics
        exp_data['final_metrics'] = get_final_metrics(exp_data['history'])
        
        experiments_data.append(exp_data)
    
    # Generate report
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LoRA FINE-TUNING EXPERIMENTS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Experiments: {len(experiments_data)}\n")
        f.write(f"Experiments Directory: {experiments_dir}\n\n")
        
        # Summary table
        f.write("=" * 80 + "\n")
        f.write("RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Header
        f.write(f"{'Experiment':<25} {'LoRA r':<8} {'α':<8} {'LR':<10} {'Final Loss':<12} {'Best Loss':<12}\n")
        f.write("-" * 80 + "\n")
        
        # Sort by best eval loss
        sorted_exps = sorted(experiments_data, 
                           key=lambda x: x['final_metrics']['best_eval_loss'] 
                           if x['final_metrics']['best_eval_loss'] is not None else float('inf'))
        
        for exp in sorted_exps:
            name = exp['name']
            config = exp['config']
            metrics = exp['final_metrics']
            
            lora_r = config.get('LoRA r', 'N/A')
            lora_alpha = config.get('LoRA alpha', 'N/A')
            lr = config.get('Learning rate', 'N/A')
            
            final_loss = f"{metrics['final_eval_loss']:.4f}" if metrics['final_eval_loss'] else "N/A"
            best_loss = f"{metrics['best_eval_loss']:.4f}" if metrics['best_eval_loss'] else "N/A"
            
            f.write(f"{name:<25} {lora_r:<8} {lora_alpha:<8} {lr:<10} {final_loss:<12} {best_loss:<12}\n")
        
        f.write("\n")
        
        # Detailed results for each experiment
        f.write("=" * 80 + "\n")
        f.write("DETAILED EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for exp in experiments_data:
            f.write("-" * 80 + "\n")
            f.write(f"Experiment: {exp['name']}\n")
            f.write("-" * 80 + "\n")
            
            # Configuration
            f.write("\nConfiguration:\n")
            for key, value in exp['config'].items():
                f.write(f"  {key}: {value}\n")
            
            # Final metrics
            f.write("\nFinal Metrics:\n")
            metrics = exp['final_metrics']
            if metrics['final_train_loss']:
                f.write(f"  Final Train Loss: {metrics['final_train_loss']:.4f}\n")
            if metrics['final_eval_loss']:
                f.write(f"  Final Eval Loss: {metrics['final_eval_loss']:.4f}\n")
            if metrics['best_eval_loss']:
                f.write(f"  Best Eval Loss: {metrics['best_eval_loss']:.4f}\n")
            if metrics['total_steps']:
                f.write(f"  Total Steps: {metrics['total_steps']}\n")
            
            f.write("\n")
        
        # Best performing configurations
        f.write("=" * 80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 80 + "\n\n")
        
        if sorted_exps and sorted_exps[0]['final_metrics']['best_eval_loss']:
            best_exp = sorted_exps[0]
            f.write("Best Performing Configuration:\n")
            f.write(f"  Experiment: {best_exp['name']}\n")
            f.write(f"  Best Eval Loss: {best_exp['final_metrics']['best_eval_loss']:.4f}\n")
            f.write(f"  Configuration:\n")
            for key, value in best_exp['config'].items():
                f.write(f"    {key}: {value}\n")
            f.write("\n")
        
        # Analysis
        f.write("Analysis:\n")
        
        # Compare by LoRA rank
        rank_performance = {}
        for exp in experiments_data:
            rank = exp['config'].get('LoRA r', 'N/A')
            loss = exp['final_metrics']['best_eval_loss']
            if loss and rank != 'N/A':
                if rank not in rank_performance:
                    rank_performance[rank] = []
                rank_performance[rank].append(loss)
        
        if rank_performance:
            f.write("\n  Performance by LoRA Rank:\n")
            for rank, losses in sorted(rank_performance.items()):
                avg_loss = sum(losses) / len(losses)
                f.write(f"    r={rank}: avg loss = {avg_loss:.4f} ({len(losses)} experiments)\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nReport generated successfully: {output_file}")


def main():
    args = parse_args()
    generate_report(args.experiments_dir, args.output_file)


if __name__ == "__main__":
    main()
