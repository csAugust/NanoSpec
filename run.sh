CUDA_VISIBLE_DEVICES=7 python examples/example_generate.py

# rebuild cuda
DEBUG_BUILD=1 python setup.py build_ext --inplace

# 1. Run evaluations
bash scripts/spec_bench/llama3-8b-instruct/run_baseline.sh
bash scripts/spec_bench/llama3-8b-instruct/run_eagle.sh
bash scripts/spec_bench/llama3-8b-instruct/run_eagle_fr_spec.sh

# 2. Evaluate speed
bash scripts/spec_bench/llama3-8b-instruct/speed_up.sh

# 3. Check correctness (for human_eval and gsm8k only)
bash scripts/spec_bench/llama3-8b-instruct/check_correctness.sh