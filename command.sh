PYTHONPATH="$PWD" CODESTRAL_API_KEY=MIWBZgTZeKQgM5xWcwDuZFtBWeEi2mbR conda run -n openevolve_test \
    python scripts/run_benchmark_matrix.py \
    --runs 5 \
    --benchmarks func,cp,sp,kmod \
    --strategies mcts,baseline \
    --output-root output/benchmark_matrix_full_4x5_ver1 \
    --max-total-tokens 5000000 \
    --fail-fast