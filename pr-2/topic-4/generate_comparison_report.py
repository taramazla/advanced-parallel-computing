"""
Script untuk generate comparison report dari hasil eksperimen CPU vs GPU
"""

import argparse
import pandas as pd
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CPU vs GPU comparison report")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="Path to timing results CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for report")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path")
    return parser.parse_args()


def format_time(seconds):
    """Format seconds to human readable format"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}m {secs}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"


def calculate_speedup(cpu_time, gpu_time):
    """Calculate speedup factor"""
    if gpu_time > 0:
        return cpu_time / gpu_time
    return 0


def generate_report(df, output_dir):
    """Generate markdown report"""
    report_path = os.path.join(output_dir, "comparison_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# CPU vs GPU Training Comparison Report\n\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        
        cpu_experiments = df[df['device'] == 'CPU']
        gpu_experiments = df[df['device'] == 'GPU']
        
        if len(cpu_experiments) > 0:
            f.write("### CPU Training\n\n")
            f.write(f"- Total experiments: {len(cpu_experiments)}\n")
            f.write(f"- Average duration: {format_time(int(cpu_experiments['duration_seconds'].mean()))}\n")
            f.write(f"- Fastest: {format_time(int(cpu_experiments['duration_seconds'].min()))}\n")
            f.write(f"- Slowest: {format_time(int(cpu_experiments['duration_seconds'].max()))}\n\n")
        
        if len(gpu_experiments) > 0:
            f.write("### GPU Training\n\n")
            f.write(f"- Total experiments: {len(gpu_experiments)}\n")
            f.write(f"- Average duration: {format_time(int(gpu_experiments['duration_seconds'].mean()))}\n")
            f.write(f"- Fastest: {format_time(int(gpu_experiments['duration_seconds'].min()))}\n")
            f.write(f"- Slowest: {format_time(int(gpu_experiments['duration_seconds'].max()))}\n\n")
            
            # Calculate average speedup
            if len(cpu_experiments) > 0:
                avg_cpu = cpu_experiments['duration_seconds'].mean()
                avg_gpu = gpu_experiments['duration_seconds'].mean()
                speedup = calculate_speedup(avg_cpu, avg_gpu)
                f.write(f"### Performance Gain\n\n")
                f.write(f"- **Average Speedup: {speedup:.2f}x**\n")
                f.write(f"- GPU is {speedup:.2f} times faster than CPU on average\n\n")
        
        # Detailed Results Table
        f.write("## Detailed Results\n\n")
        f.write("| Experiment | Device | Batch Size | Grad Accum | Quantization | Duration | Effective Batch |\n")
        f.write("|------------|--------|------------|------------|--------------|----------|----------------|\n")
        
        for _, row in df.iterrows():
            effective_batch = row['batch_size'] * row['grad_accum']
            f.write(f"| {row['experiment']} | {row['device']} | {row['batch_size']} | "
                   f"{row['grad_accum']} | {row['quantization']} | "
                   f"{format_time(int(row['duration_seconds']))} | {effective_batch} |\n")
        
        # Configuration Comparisons
        f.write("\n## Configuration Analysis\n\n")
        
        # Batch size impact
        f.write("### Batch Size Impact\n\n")
        for device in df['device'].unique():
            device_df = df[df['device'] == device]
            if len(device_df) > 1:
                f.write(f"**{device}:**\n\n")
                for _, row in device_df.iterrows():
                    eff_batch = row['batch_size'] * row['grad_accum']
                    f.write(f"- Batch {row['batch_size']} x Accum {row['grad_accum']} "
                           f"(Effective: {eff_batch}): {format_time(int(row['duration_seconds']))}\n")
                f.write("\n")
        
        # Quantization impact (GPU only)
        if len(gpu_experiments) > 0:
            f.write("### Quantization Impact (GPU)\n\n")
            quant_groups = gpu_experiments.groupby('quantization')['duration_seconds'].mean()
            for quant_type, avg_time in quant_groups.items():
                f.write(f"- {quant_type}: {format_time(int(avg_time))} (average)\n")
            f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if len(gpu_experiments) > 0 and len(cpu_experiments) > 0:
            f.write("### For Production Training\n\n")
            
            fastest_gpu = gpu_experiments.loc[gpu_experiments['duration_seconds'].idxmin()]
            f.write(f"**Recommended GPU Configuration:**\n")
            f.write(f"- Batch size: {fastest_gpu['batch_size']}\n")
            f.write(f"- Gradient accumulation: {fastest_gpu['grad_accum']}\n")
            f.write(f"- Quantization: {fastest_gpu['quantization']}\n")
            f.write(f"- Expected duration: {format_time(int(fastest_gpu['duration_seconds']))}\n\n")
        
        if len(cpu_experiments) > 0:
            fastest_cpu = cpu_experiments.loc[cpu_experiments['duration_seconds'].idxmin()]
            f.write(f"**Recommended CPU Configuration (for testing):**\n")
            f.write(f"- Batch size: {fastest_cpu['batch_size']}\n")
            f.write(f"- Gradient accumulation: {fastest_cpu['grad_accum']}\n")
            f.write(f"- Expected duration: {format_time(int(fastest_cpu['duration_seconds']))}\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if len(gpu_experiments) > 0 and len(cpu_experiments) > 0:
            avg_speedup = calculate_speedup(
                cpu_experiments['duration_seconds'].mean(),
                gpu_experiments['duration_seconds'].mean()
            )
            f.write(f"GPU training provides approximately **{avg_speedup:.1f}x speedup** over CPU training "
                   f"for this workload. ")
            
            if avg_speedup >= 5:
                f.write("This significant speedup makes GPU essential for production training.\n\n")
            elif avg_speedup >= 2:
                f.write("GPU provides meaningful acceleration, recommended for iterative development.\n\n")
            else:
                f.write("Speedup is modest, CPU may be sufficient for small-scale experiments.\n\n")
        elif len(cpu_experiments) > 0:
            f.write("Only CPU experiments were conducted. GPU experiments require CUDA-enabled hardware.\n\n")
        
        f.write("**Key Findings:**\n\n")
        f.write("1. Larger batch sizes generally improve GPU utilization\n")
        f.write("2. Gradient accumulation allows effective larger batches with limited memory\n")
        f.write("3. Quantization (4-bit/8-bit) enables training larger models on consumer GPUs\n")
        f.write("4. CPU training is viable for testing but impractical for production-scale training\n")
    
    return report_path


def main():
    args = parse_args()
    
    # Read results
    if not os.path.exists(args.results_csv):
        print(f"Error: Results file not found: {args.results_csv}")
        return
    
    df = pd.read_csv(args.results_csv)
    
    if len(df) == 0:
        print("No results to process")
        return
    
    print(f"Processing {len(df)} experiments...")
    
    # Generate report
    report_path = generate_report(df, args.output_dir)
    
    print(f"\n✓ Comparison report generated: {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for device in df['device'].unique():
        device_df = df[df['device'] == device]
        avg_time = device_df['duration_seconds'].mean()
        print(f"{device}: {len(device_df)} experiments, avg time: {format_time(int(avg_time))}")
    
    # Calculate speedup if both CPU and GPU results exist
    cpu_df = df[df['device'] == 'CPU']
    gpu_df = df[df['device'] == 'GPU']
    
    if len(cpu_df) > 0 and len(gpu_df) > 0:
        speedup = calculate_speedup(
            cpu_df['duration_seconds'].mean(),
            gpu_df['duration_seconds'].mean()
        )
        print(f"\nGPU Speedup: {speedup:.2f}x faster than CPU")
    
    print("="*60)


if __name__ == "__main__":
    main()
