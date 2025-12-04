#!/usr/bin/env python3
"""
Analysis and visualization script for CG experiment results.
Generates plots and statistical analysis for assignment report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

def load_results(results_dir="experiment_results"):
    """Load experiment results from CSV files."""
    results_path = Path(results_dir)

    raw_csv = results_path / "cg_results.csv"
    summary_csv = results_path / "cg_summary.csv"

    if not raw_csv.exists():
        print(f"Error: {raw_csv} not found. Run run_experiments.sh first.")
        sys.exit(1)

    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else None

    return raw_df, summary_df

def plot_speedup_vs_processors(summary_df, output_dir="experiment_results"):
    """Plot speedup vs number of processors for each problem size."""
    output_path = Path(output_dir)

    # Filter implementations that scale with processors
    parallel_impls = summary_df[summary_df['Implementation'].isin(['OpenMP_Multicore', 'MPI_Cluster'])]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Speedup vs Number of Processors', fontsize=16, fontweight='bold')

    sizes = sorted(parallel_impls['N'].unique())

    for idx, N in enumerate(sizes):
        ax = axes[idx // 2, idx % 2]

        for impl in ['OpenMP_Multicore', 'MPI_Cluster']:
            data = parallel_impls[(parallel_impls['N'] == N) & (parallel_impls['Implementation'] == impl)]
            data_sorted = data.sort_values('Processors')

            ax.plot(data_sorted['Processors'], data_sorted['Speedup'],
                   marker='o', linewidth=2, markersize=8, label=impl.replace('_', ' '))

        # Ideal speedup line
        procs = data_sorted['Processors'].values
        ax.plot(procs, procs, 'k--', linewidth=1.5, alpha=0.7, label='Ideal Speedup')

        ax.set_xlabel('Number of Processors', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(f'N = {N}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(procs)

    plt.tight_layout()
    plt.savefig(output_path / 'speedup_vs_processors.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'speedup_vs_processors.png'}")
    plt.close()

def plot_execution_time_vs_size(summary_df, output_dir="experiment_results"):
    """Plot execution time vs problem size for all implementations."""
    output_path = Path(output_dir)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Group by implementation and processors
    impls_to_plot = [
        ('CPU_Sequential', 1),
        ('OpenMP_Multicore', 8),
        ('MPI_Cluster', 8),
        ('CUDA_Custom', 1),
        ('CUDA_cuBLAS', 1),
        ('CUDA_cuSPARSE', 1)
    ]

    for impl, procs in impls_to_plot:
        data = summary_df[(summary_df['Implementation'] == impl) & (summary_df['Processors'] == procs)]
        data_sorted = data.sort_values('N')

        label = impl.replace('_', ' ')
        if procs > 1:
            label += f' ({procs}p)'

        ax.plot(data_sorted['N'], data_sorted['AvgTime'],
               marker='o', linewidth=2, markersize=8, label=label)

    ax.set_xlabel('Problem Size (N)', fontsize=13)
    ax.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax.set_title('Execution Time vs Problem Size', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)

    plt.tight_layout()
    plt.savefig(output_path / 'execution_time_vs_size.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'execution_time_vs_size.png'}")
    plt.close()

def plot_efficiency(summary_df, output_dir="experiment_results"):
    """Plot parallel efficiency for OpenMP and MPI."""
    output_path = Path(output_dir)

    parallel_impls = summary_df[summary_df['Implementation'].isin(['OpenMP_Multicore', 'MPI_Cluster'])]

    # Calculate efficiency = Speedup / Processors
    parallel_impls = parallel_impls.copy()
    parallel_impls['Efficiency'] = parallel_impls['Speedup'] / parallel_impls['Processors']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Parallel Efficiency (Speedup / Processors)', fontsize=16, fontweight='bold')

    for idx, impl in enumerate(['OpenMP_Multicore', 'MPI_Cluster']):
        ax = axes[idx]
        data = parallel_impls[parallel_impls['Implementation'] == impl]

        for N in sorted(data['N'].unique()):
            subset = data[data['N'] == N].sort_values('Processors')
            ax.plot(subset['Processors'], subset['Efficiency'] * 100,
                   marker='o', linewidth=2, markersize=8, label=f'N={N}')

        # Ideal efficiency line
        ax.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal (100%)')

        ax.set_xlabel('Number of Processors', fontsize=12)
        ax.set_ylabel('Efficiency (%)', fontsize=12)
        ax.set_title(impl.replace('_', ' '), fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(output_path / 'parallel_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'parallel_efficiency.png'}")
    plt.close()

def plot_gpu_comparison(summary_df, output_dir="experiment_results"):
    """Compare different GPU implementations."""
    output_path = Path(output_dir)

    gpu_impls = summary_df[summary_df['Implementation'].str.contains('CUDA')]

    fig, ax = plt.subplots(figsize=(10, 6))

    for impl in gpu_impls['Implementation'].unique():
        data = gpu_impls[gpu_impls['Implementation'] == impl].sort_values('N')
        ax.plot(data['N'], data['AvgTime'],
               marker='o', linewidth=2, markersize=8, label=impl.replace('CUDA_', ''))

    ax.set_xlabel('Problem Size (N)', fontsize=13)
    ax.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax.set_title('GPU Implementation Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_path / 'gpu_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'gpu_comparison.png'}")
    plt.close()

def generate_latex_tables(summary_df, output_dir="experiment_results"):
    """Generate LaTeX tables for assignment report."""
    output_path = Path(output_dir)

    # Table 1: Best configurations for each implementation
    table1 = summary_df.loc[summary_df.groupby(['Implementation', 'N'])['AvgTime'].idxmin()]
    table1_sorted = table1.sort_values(['N', 'AvgTime'])

    with open(output_path / 'table_best_configs.tex', 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Best Performance Configurations}\n")
        f.write("\\begin{tabular}{llrrrr}\n")
        f.write("\\hline\n")
        f.write("Implementation & N & Procs & Avg Time (s) & Speedup & Efficiency \\\\\n")
        f.write("\\hline\n")

        for _, row in table1_sorted.iterrows():
            efficiency = (row['Speedup'] / row['Processors'] * 100) if row['Processors'] > 1 else 100
            f.write(f"{row['Implementation'].replace('_', ' ')} & {int(row['N'])} & {int(row['Processors'])} & ")
            f.write(f"{row['AvgTime']:.4f} & {row['Speedup']:.2f} & {efficiency:.1f}\\% \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"Saved: {output_path / 'table_best_configs.tex'}")

    # Table 2: Speedup comparison for N=8192
    if 8192 in summary_df['N'].values:
        table2 = summary_df[summary_df['N'] == 8192].sort_values('AvgTime')

        with open(output_path / 'table_speedup_8192.tex', 'w') as f:
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison for N=8192}\n")
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\hline\n")
            f.write("Implementation & Procs & Time (s) & Speedup & Iterations \\\\\n")
            f.write("\\hline\n")

            for _, row in table2.iterrows():
                f.write(f"{row['Implementation'].replace('_', ' ')} & {int(row['Processors'])} & ")
                f.write(f"{row['AvgTime']:.4f} & {row['Speedup']:.2f} & {int(row['AvgIterations'])} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"Saved: {output_path / 'table_speedup_8192.tex'}")

def generate_analysis_report(raw_df, summary_df, output_dir="experiment_results"):
    """Generate text analysis report."""
    output_path = Path(output_dir)

    with open(output_path / 'analysis_report.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CONJUGATE GRADIENT PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        f.write("1. OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total experiments: {len(raw_df)}\n")
        f.write(f"Implementations tested: {summary_df['Implementation'].nunique()}\n")
        f.write(f"Problem sizes: {sorted(summary_df['N'].unique())}\n")
        f.write(f"Processor counts: {sorted(summary_df['Processors'].unique())}\n\n")

        # Best performers
        f.write("2. BEST PERFORMERS\n")
        f.write("-" * 70 + "\n")

        for N in sorted(summary_df['N'].unique()):
            best = summary_df[summary_df['N'] == N].nsmallest(1, 'AvgTime').iloc[0]
            f.write(f"N={N}:\n")
            f.write(f"  Implementation: {best['Implementation']}\n")
            f.write(f"  Processors: {int(best['Processors'])}\n")
            f.write(f"  Time: {best['AvgTime']:.4f} seconds\n")
            f.write(f"  Speedup: {best['Speedup']:.2f}x\n\n")

        # Speedup analysis
        f.write("3. SPEEDUP ANALYSIS\n")
        f.write("-" * 70 + "\n")

        max_speedup = summary_df.nlargest(1, 'Speedup').iloc[0]
        f.write(f"Maximum speedup: {max_speedup['Speedup']:.2f}x\n")
        f.write(f"  Implementation: {max_speedup['Implementation']}\n")
        f.write(f"  Problem size: N={int(max_speedup['N'])}\n")
        f.write(f"  Processors: {int(max_speedup['Processors'])}\n\n")

        # GPU vs MPI vs OpenMP
        f.write("4. GPU vs MPI vs OPENMP (N=8192, 8 processors)\n")
        f.write("-" * 70 + "\n")

        if 8192 in summary_df['N'].values:
            gpu_best = summary_df[(summary_df['N'] == 8192) &
                                 (summary_df['Implementation'].str.contains('CUDA'))].nsmallest(1, 'AvgTime').iloc[0]
            mpi = summary_df[(summary_df['N'] == 8192) &
                            (summary_df['Implementation'] == 'MPI_Cluster') &
                            (summary_df['Processors'] == 8)].iloc[0] if len(summary_df[(summary_df['N'] == 8192) &
                            (summary_df['Implementation'] == 'MPI_Cluster') &
                            (summary_df['Processors'] == 8)]) > 0 else None
            omp = summary_df[(summary_df['N'] == 8192) &
                            (summary_df['Implementation'] == 'OpenMP_Multicore') &
                            (summary_df['Processors'] == 8)].iloc[0] if len(summary_df[(summary_df['N'] == 8192) &
                            (summary_df['Implementation'] == 'OpenMP_Multicore') &
                            (summary_df['Processors'] == 8)]) > 0 else None

            f.write(f"GPU (best): {gpu_best['AvgTime']:.4f}s (speedup: {gpu_best['Speedup']:.2f}x)\n")
            if mpi is not None:
                f.write(f"MPI (8p): {mpi['AvgTime']:.4f}s (speedup: {mpi['Speedup']:.2f}x)\n")
            if omp is not None:
                f.write(f"OpenMP (8t): {omp['AvgTime']:.4f}s (speedup: {omp['Speedup']:.2f}x)\n\n")

        # Convergence
        f.write("5. CONVERGENCE ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Average iterations across all experiments: {summary_df['AvgIterations'].mean():.1f}\n")
        f.write(f"Std dev: {summary_df['AvgIterations'].std():.1f}\n")
        f.write(f"Note: All implementations should converge to same number of iterations\n")
        f.write(f"      for the same problem (same random seed).\n\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"Saved: {output_path / 'analysis_report.txt'}")

def main():
    print("=" * 70)
    print("CG EXPERIMENT ANALYSIS AND VISUALIZATION")
    print("=" * 70)
    print()

    # Load data
    print("Loading results...")
    raw_df, summary_df = load_results()

    if summary_df is None:
        print("Warning: Summary CSV not found. Generating from raw data...")
        # Could implement summary generation here if needed
        print("Please run run_experiments.sh to generate summary.")
        return

    print(f"Loaded {len(raw_df)} raw results and {len(summary_df)} summary entries.\n")

    # Generate plots
    print("Generating plots...")
    plot_speedup_vs_processors(summary_df)
    plot_execution_time_vs_size(summary_df)
    plot_efficiency(summary_df)
    plot_gpu_comparison(summary_df)
    print()

    # Generate LaTeX tables
    print("Generating LaTeX tables...")
    generate_latex_tables(summary_df)
    print()

    # Generate text report
    print("Generating analysis report...")
    generate_analysis_report(raw_df, summary_df)
    print()

    print("=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - speedup_vs_processors.png")
    print("  - execution_time_vs_size.png")
    print("  - parallel_efficiency.png")
    print("  - gpu_comparison.png")
    print("  - table_best_configs.tex")
    print("  - table_speedup_8192.tex")
    print("  - analysis_report.txt")
    print("\nUse these for your assignment report!")

if __name__ == "__main__":
    main()
