import random
import csv
import os
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
import pandas as pd
from utils import SYSTEM_PROMPT_OURS, PROMPT_OURS, PROMPT_ROBUST


LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

random.seed(0)


def get_llm_outputs(llm, prompts, sampling_params, system_prompt):
    prompts1 = [[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}] 
                for prompt in prompts]
    tokenizer = llm.get_tokenizer()
    prompts2 = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts1]
    llm_outputs = llm.generate(prompts2, sampling_params)
    llm_outputs = [output.outputs[0].text.strip() for output in llm_outputs]
    llm_outputs = [output.replace(prompt, "").strip() for prompt, output in zip(prompts, llm_outputs)]

    return llm_outputs



def init_llm():
    llm = LLM("Qwen/Qwen2.5-72B-Instruct", tensor_parallel_size=4, swap_space=4)
    llm_tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens = 256)
    sampling_params.stop = [llm_tokenizer.eos_token, "<|eot_id|>"]
    return llm, sampling_params

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, help = "model path")
    parser.add_argument("--dataset", type = str, default = "factual_recall", help = "incontext_recall/factual_recall/counter_factual")
    args = parser.parse_args()
    if args.dataset == "incontext_factual":
        PROMPT_OURS = PROMPT_ROBUST
    llm, sampling_params = init_llm()
    args.model = os.path.split(args.model)[1]
    prompt = "without_system_prompt"
    data_scores = {}
    args.prompt = prompt
    data_scores[prompt] = []
    for lang in LANGUAGES:
        data = pd.read_csv(f"generations/{args.dataset}/{args.model}/{args.prompt}/{lang}.csv")
        prompts = []

        for i in range(len(data)):
            prompts.append(PROMPT_OURS.format(question = data.iloc[i]["Question"], predicted = data.iloc[i]["llm_output"], answers = data.iloc[i]["answers"]))

        llm_outputs = get_llm_outputs(llm, prompts, sampling_params, SYSTEM_PROMPT_OURS)
        results = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(llm_outputs)):
            if "[1]" in llm_outputs[i]:
                results[1] += 1
            elif "[2]" in llm_outputs[i]:
                if lang == "English":
                    results[1] += 1
                else:
                    results[2] += 1
            elif "[3]" in llm_outputs[i]:
                results[3] += 1
            else:
                results[4] += 1
        data_scores[prompt].append(results)
        if not os.path.exists(f"results/{args.dataset}/{args.model}/{args.prompt}"):
            os.makedirs(f"results/{args.dataset}/{args.model}/{args.prompt}")
        with open(f"results/{args.dataset}/{args.model}/{args.prompt}/{lang}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "Predicted", "answers", "label", "llm_output"])
            for i in range(len(data)):
                if "label" in data.columns:
                    label = data.iloc[i]["label"]
                else:
                    label = ""
                writer.writerow([data.iloc[i]["Question"], data.iloc[i]["llm_output"], data.iloc[i]["answers"], label, llm_outputs[i]])
    types = ["Same-Correct", "English-Correct", "Different-Correct", "Incorrect"]
    for i in range(1, 5):
        with open(f"results/{args.dataset}/{args.model}/results_exact_match_{args.prompt}.csv","a") as f:
            writer = csv.writer(f)
            if i == 1:
                writer.writerow(["Type"] + LANGUAGES)
            scores = []
            for j in range(len(LANGUAGES)):
                scores.append(data_scores[prompt][j][i])
            writer.writerow([types[i-1]] + scores)

    





