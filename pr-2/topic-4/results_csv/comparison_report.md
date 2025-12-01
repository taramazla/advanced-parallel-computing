# CPU vs GPU Training Comparison Report

Generated at: 2025-12-01 21:26:19

## Summary Statistics

### CPU Training

- Total experiments: 3
- Average duration: 18m 52s
- Fastest: 9m 32s
- Slowest: 30m 57s

### GPU Training

- Total experiments: 3
- Average duration: 10m 41s
- Fastest: 6m 41s
- Slowest: 16m 13s

### Performance Gain

- **Average Speedup: 1.77x**
- GPU is 1.77 times faster than CPU on average

## Detailed Results

| Experiment | Device | Batch Size | Grad Accum | Quantization | Duration | Effective Batch |
|------------|--------|------------|------------|--------------|----------|----------------|
| cpu_bs2_ga4 | CPU | 2 | 4 | none | 16m 8s | 8 |
| cpu_bs1_ga8 | CPU | 1 | 8 | none | 30m 57s | 8 |
| cpu_bs4_ga2 | CPU | 4 | 2 | none | 9m 32s | 8 |
| gpu_bs2_ga4 | GPU | 2 | 4 | none | 16m 13s | 8 |
| gpu_bs4_ga2 | GPU | 4 | 2 | none | 9m 10s | 8 |
| gpu_bs8_ga1 | GPU | 8 | 1 | none | 6m 41s | 8 |

## Configuration Analysis

### Batch Size Impact

**CPU:**

- Batch 2 x Accum 4 (Effective: 8): 16m 8s
- Batch 1 x Accum 8 (Effective: 8): 30m 57s
- Batch 4 x Accum 2 (Effective: 8): 9m 32s

**GPU:**

- Batch 2 x Accum 4 (Effective: 8): 16m 13s
- Batch 4 x Accum 2 (Effective: 8): 9m 10s
- Batch 8 x Accum 1 (Effective: 8): 6m 41s

### Quantization Impact (GPU)

- none: 10m 41s (average)

## Recommendations

### For Production Training

**Recommended GPU Configuration:**
- Batch size: 8
- Gradient accumulation: 1
- Quantization: none
- Expected duration: 6m 41s

**Recommended CPU Configuration (for testing):**
- Batch size: 4
- Gradient accumulation: 2
- Expected duration: 9m 32s

## Conclusion

GPU training provides approximately **1.8x speedup** over CPU training for this workload. Speedup is modest, CPU may be sufficient for small-scale experiments.

**Key Findings:**

1. Larger batch sizes generally improve GPU utilization
2. Gradient accumulation allows effective larger batches with limited memory
3. Quantization (4-bit/8-bit) enables training larger models on consumer GPUs
4. CPU training is viable for testing but impractical for production-scale training
