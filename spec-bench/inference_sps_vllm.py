"""Generate answers using vLLM server with messages API.
"""
import argparse
import json
import time
import requests
from openai import OpenAI
from transformers import AutoTokenizer

from eval_vllm import run_eval, reorg_answer_file

def get_metrics(url="http://0.0.0.0:8000/metrics"):
    """Fetch metrics from vLLM server."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        metrics = response.text
        return metrics
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metrics: {e}")
        return None

def parse_draft_acceptance_rate(metrics):
    """Parse the draft acceptance rate from vLLM metrics."""
    metric = -1
    if metrics:
        for line in metrics.splitlines():
            if line.startswith("vllm:spec_decode_draft_acceptance_rate"):
                metric = float(line.split()[-1])
    else:
        print("No metrics to parse")
    return metric

def vllm_forward(messages, model_id, max_new_tokens, tokenizer, api_base="http://localhost:8000/v1", 
                metrics_url="http://0.0.0.0:8000/metrics", temperature=0.0, **kwargs):
    """Forward function for vLLM server using messages API."""
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base,
    )
    
    # Store initial metrics to compare later
    initial_metrics = get_metrics(metrics_url)
    
    # Measure start time
    start_time = time.time()
    
    try:
        # Create chat completion with messages
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            n=1,
            stream=False,
        )
        output_text = completion.choices[0].message.content
        
        # Get token count from the API response if available
        if hasattr(completion, 'usage') and completion.usage and hasattr(completion.usage, 'completion_tokens'):
            new_token_count = completion.usage.completion_tokens
        else:
            # Use the tokenizer to get accurate token count
            output_ids = tokenizer.encode(output_text)
            new_token_count = len(output_ids)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        output_text = "ERROR"
        # Use tokenizer to count tokens for "ERROR"
        output_ids = tokenizer.encode("ERROR")
        new_token_count = len(output_ids)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Get metrics after generation
    post_metrics = get_metrics(metrics_url)
    draft_acceptance_rate = parse_draft_acceptance_rate(post_metrics)
    
    # Create accept_length_tree from draft acceptance rate
    # This is an approximation as we don't have the actual length list
    accept_length_tree = [int(new_token_count * draft_acceptance_rate)] * new_token_count if new_token_count > 0 else []
    
    # Return values (output text, token count, step count, acceptance metrics)
    step_count = 1  # Placeholder since we don't have actual step count
    
    return output_text, new_token_count, step_count, accept_length_tree, total_time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate answers using vLLM server")
    
    # Model and server configuration
    parser.add_argument(
        "--model-id", 
        type=str, 
        required=True,
        help="Model ID to use on the vLLM server"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to the tokenizer for token counting. If not provided, will use model-id.",
    )
    parser.add_argument(
        "--api-base", 
        type=str, 
        default="http://localhost:8000/v1",
        help="Base URL for vLLM server API"
    )
    parser.add_argument(
        "--metrics-url", 
        type=str, 
        default="http://0.0.0.0:8000/metrics",
        help="URL for vLLM metrics endpoint"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end",
        type=int,
        help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--answer-file", 
        type=str, 
        help="The output answer file."
    )
    
    # Generation parameters
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    
    # Deprecated parameters (kept for backwards compatibility)
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="[DEPRECATED] Kept for compatibility but not used.",
    )
    parser.add_argument(
        "--num-gpus-total", 
        type=int, 
        default=1, 
        help="[DEPRECATED] Kept for compatibility but not used."
    )
    
    return parser.parse_args()

def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    # Set up file paths
    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    # Load tokenizer for token counting
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_id
    try:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Falling back to default tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fallback to a common tokenizer

    # Set do_sample based on temperature
    do_sample = args.temperature > 0

    # Run evaluation
    run_eval(
        forward_func=vllm_forward,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        api_base=args.api_base,
        metrics_url=args.metrics_url,
        temperature=args.temperature,
        do_sample=do_sample,
        tokenizer=tokenizer,
    )

    # Reorganize the answer file
    reorg_answer_file(answer_file)

if __name__ == "__main__":
    main()