#!/usr/bin/env bash
set -euo pipefail


# Time limit in seconds for one benchmark script. The slowest one (mhc) takes about 45 minutes.
BENCHMARK_TIMEOUT="${BENCHMARK_TIMEOUT:-3600}"

# Array to keep track of failed benchmarks
FAILED_BENCHMARKS=()

for file in Liger-Kernel/benchmark/scripts/benchmark_*; do
    # TODO: unskip when https://github.com/intel/intel-xpu-backend-for-triton/issues/3873 is resolved
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_orpo_loss.py" ] || [ $file = "Liger-Kernel/benchmark/scripts/benchmark_grpo_loss.py" ]; then
      continue
    fi
    # TODO: unskip when https://github.com/intel/intel-xpu-backend-for-triton/issues/5826 is resolved
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_fused_linear_jsd.py" ]; then
        continue
    fi
    # TODO: unskip when https://github.com/intel/intel-xpu-backend-for-triton/issues/8737 is resolved
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_sparse_multi_token_attention.py" ] || [ $file = "Liger-Kernel/benchmark/scripts/benchmark_multi_token_attention.py" ]; then
        continue
    fi
    # TODO: file an issue and unskip when it is resolved: the script calls torch.cuda.synchronize()
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_fused_moe.py" ]; then
        continue
    fi
    # TODO: file an issue and unskip when it is resolved: fp32 nn.LayerNorm is called with a bf16 input
    if [ $file = "Liger-Kernel/benchmark/scripts/benchmark_layer_norm.py" ]; then
        continue
    fi
    # A hung benchmark must not block the remaining ones until the job timeout.
    if timeout -k 60 "$BENCHMARK_TIMEOUT" python "$file"; then
        echo "Benchmark ran successfully: $file"
    else
        echo "Error: Benchmark failed for $file."
        FAILED_BENCHMARKS+=("$file")
    fi
done

# Print failed benchmarks
if [ ${#FAILED_BENCHMARKS[@]} -ne 0 ]; then
    echo "The following benchmarks failed:"
    for failed_bench in "${FAILED_BENCHMARKS[@]}"; do
        echo "$failed_bench"
    done
    exit 1
else
    echo "All benchmarks completed successfully."
fi

exit 0
