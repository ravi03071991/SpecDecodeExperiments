# Speculative Decoding Benchmarking with vLLM

This repo demonstrates speculative decoding using vLLM with Qwen models.

## Start vLLM API Server

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --seed 42 -tp 1 --gpu_memory_utilization 0.8 \
  --speculative_model "Qwen/Qwen2.5-0.5B-Instruct" \
  --num_speculative_tokens 5
```

## Run Inference Benchmark

```bash
python inference_sps_vllm.py \
  --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
  --tokenizer-path "Qwen/Qwen2.5-1.5B-Instruct" \
  --api-base "http://localhost:8000/v1" \
  --metrics-url "http://0.0.0.0:8000/metrics" \
  --bench-name "spec_bench" \
  --answer-file "output/answers.jsonl" \
  --max-new-tokens 1024 \
  --num-choices 1
```