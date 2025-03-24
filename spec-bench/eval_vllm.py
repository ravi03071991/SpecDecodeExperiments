"""Generate answers with vLLM server using messages API.
"""
# adapted from fastchat: https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/gen_model_answer.py

import json
import os
import time
import numpy as np
import shortuuid

# from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from typing import Optional

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def run_eval(
        forward_func,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_tokens,
        num_choices,
        num_gpus_per_model=1,
        num_gpus_total=1,
        **kwargs,
):
    questions = load_questions(question_file, question_begin, question_end)

    # Process all questions directly without Ray
    get_model_answers(
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
    )


def get_model_answers(
        forward_func,
        model_id,
        questions,
        answer_file,
        max_new_tokens,
        num_choices,
        **kwargs,
):
    print('Starting evaluation with vLLM server')
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup with first question
    print('Starting warmup...')
    for _ in range(3):
        messages = []
        for turn in question["turns"]:
            # Add user message
            messages.append({
                "role": "user",
                "content": turn
            })
            
            try:
                # Call vLLM forward function with the messages
                output_text, new_token, step, accept_length_tree, total_time = forward_func(
                    messages,
                    model_id,
                    max_new_tokens,
                    **kwargs,
                )
                
                # Add assistant message for next turn
                messages.append({
                    "role": "assistant",
                    "content": output_text
                })
                
            except Exception as e:
                print(f"ERROR during warmup: {e}")
                print("ERROR question ID: ", question["question_id"])
                output_text = "ERROR"
                
                # Add error as assistant message for next turn
                messages.append({
                    "role": "assistant",
                    "content": output_text
                })
            
    print('Warmup done')

    # Actual evaluation
    accept_lengths_tree = []
    for question in tqdm(questions):
        choices = []
        for i in range(num_choices):
            np.random.seed(i)  # Use numpy random for seed
            
            messages = []
            turns = []
            steps = []
            new_tokens = []
            wall_time = []
            cur_accept_lengths_tree = []
            
            for turn in question["turns"]:
                # Add user message
                messages.append({
                    "role": "user",
                    "content": turn
                })
                
                try:
                    # Call vLLM forward function with the messages
                    output_text, new_token, step, accept_length_tree, total_time = forward_func(
                        messages,
                        model_id,
                        max_new_tokens,
                        **kwargs,
                    )
                    
                    # Add assistant message for next turn
                    messages.append({
                        "role": "assistant",
                        "content": output_text
                    })
                    
                    # Track acceptance metrics
                    accept_lengths_tree.extend(accept_length_tree)
                    cur_accept_lengths_tree.extend(accept_length_tree)
                    
                except Exception as e:
                    print(f"ERROR during generation: {e}")
                    print("ERROR question ID: ", question["question_id"])
                    output_text = "ERROR"
                    new_token = 0
                    step = 0
                    accept_length_tree = []
                    total_time = 0
                    
                    # Add error as assistant message for next turn
                    messages.append({
                        "role": "assistant",
                        "content": output_text
                    })

                turns.append(output_text)
                steps.append(int(step))
                new_tokens.append(int(new_token))
                wall_time.append(float(total_time))
                
            # Add this choice to the choices list
            choices.append({
                "index": i, 
                "turns": turns, 
                "decoding_steps": steps, 
                "new_tokens": new_tokens, 
                "wall_time": wall_time,
                "accept_lengths": cur_accept_lengths_tree
            })

        # Dump answers for this question
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
            
    # Print overall metrics
    if accept_lengths_tree:
        print("#Mean accepted tokens: ", np.mean(accept_lengths_tree))
    else:
        print("No acceptance data collected")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])