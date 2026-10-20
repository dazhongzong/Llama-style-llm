# SmolLM2-135M-Instruct Decode Benchmark

## Environment

- Model: `SmolLM2-135M-Instruct`
- GPU: `NVIDIA GeForce RTX 5070`
- Benchmark: single-token incremental decode
- Prefill is not included in the timed region.
- Benchmark executable: `bench_transformer_decode`

## Results

| Context | Samples | CPU Time | CPU Noise | GPU Time | GPU Noise | Approx. Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 784x | 5.801 ms | 11.71% | 5.875 ms | 11.69% | 170.21 |
| 32 | 816x | 6.289 ms | 14.66% | 6.205 ms | 14.37% | 161.16 |
| 128 | 384x | 7.071 ms | 22.36% | 7.258 ms | 22.23% | 137.78 |
| 256 | 736x | 6.706 ms | 19.96% | 6.952 ms | 19.98% | 143.84 |

Approximate throughput is calculated as `1 / GPU Time` and is intended for
trend comparison only because benchmark noise is relatively high.

## Run

```bash
MINILLM_MODEL_DIR=models/SmolLM2-135M-Instruct \
    ./build-benchmarks/bench_transformer_decode \
    --benchmark model_decode
```
