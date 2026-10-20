# TinyLlama-1.1B-Chat Decode Benchmark

## Environment

- Model: `TinyLlama-1.1B-Chat-v1.0`
- GPU: `NVIDIA GeForce RTX 5070`
- Benchmark: single-token incremental decode
- Prefill is not included in the timed region.
- Benchmark executable: `bench_transformer_decode`

## Results

| Context | Samples | CPU Time | CPU Noise | GPU Time | GPU Noise | Approx. Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 144x | 10.400 ms | 8.51% | 10.660 ms | 8.57% | 93.81 |
| 32 | 80x | 10.605 ms | 9.71% | 10.841 ms | 9.74% | 92.24 |
| 128 | 80x | 10.187 ms | 9.43% | 10.442 ms | 9.53% | 95.77 |
| 256 | 320x | 11.074 ms | 9.38% | 11.353 ms | 9.39% | 88.08 |

Approximate throughput is calculated as `1 / GPU Time` and is intended for
trend comparison only.

## Run

```bash
MINILLM_MODEL_DIR=models/TinyLlama-1.1B-Chat-v1.0 \
    ./build-benchmarks/bench_transformer_decode \
    --benchmark model_decode
```
