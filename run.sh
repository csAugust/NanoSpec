source /opt/venv/bin/activate

CUDA_VISIBLE_DEVICES=7 python examples/example_generate.py

# rebuild cuda
DEBUG_BUILD=0 python setup.py build_ext --inplace

# 1. Run evaluations
bash scripts/spec_bench/llama3-8b-instruct/run_baseline.sh
bash scripts/spec_bench/llama3-8b-instruct/run_eagle.sh
bash scripts/spec_bench/llama3-8b-instruct/run_eagle_fr_spec.sh
bash scripts/spec_bench/llama3-8b-instruct/run_eagle_ours.sh

bash scripts/spec_bench/llama3.2-1b-instruct/run_eagle_fr_spec.sh
bash scripts/spec_bench/llama3.2-1b-instruct/run_eagle.sh
bash scripts/spec_bench/llama3.2-1b-instruct/run_eagle_ours.sh

bash scripts/spec_bench/qwen2-7b-instruct/run_eagle_fr_spec.sh
bash scripts/spec_bench/qwen2-7b-instruct/run_eagle.sh
bash scripts/spec_bench/qwen2-7b-instruct/run_eagle_ours.sh


bash scripts/human_eval/llama3-8b-instruct/run_baseline.sh
bash scripts/human_eval/llama3-8b-instruct/run_eagle.sh
bash scripts/human_eval/llama3-8b-instruct/run_eagle_fr_spec.sh
bash scripts/human_eval/llama3-8b-instruct/run_eagle_ours.sh

bash scripts/human_eval/llama3.2-1b-instruct/run_eagle_fr_spec.sh
bash scripts/human_eval/llama3.2-1b-instruct/run_eagle.sh
bash scripts/human_eval/llama3.2-1b-instruct/run_eagle_ours.sh

bash scripts/gsm8k/llama3-8b-instruct/run_baseline.sh
bash scripts/gsm8k/llama3-8b-instruct/run_eagle.sh
bash scripts/gsm8k/llama3-8b-instruct/run_eagle_fr_spec.sh
bash scripts/gsm8k/llama3-8b-instruct/run_eagle_ours.sh


# 2. Evaluate speed
bash scripts/spec_bench/llama3-8b-instruct/speed_up.sh

# 3. Check correctness (for human_eval and gsm8k only)
bash scripts/spec_bench/llama3-8b-instruct/check_correctness.sh