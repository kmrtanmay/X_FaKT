import random
import csv
import os
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from utils import cities, sports_persons, landmarks, politicians, festivals, artists, QUESTIONS_1, QUESTIONS_2, QUESTIONS_3, COUNTRIES


random.seed(0)

LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]




def get_llm_outputs(llm, prompts, sampling_params, system_prompt):
    if system_prompt == "":
        prompts1 = [[{"role": "user", "content": prompt}] for prompt in prompts]
    else:
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




def init_llm(model, max_tokens = 128):
    llm = LLM(model, tensor_parallel_size=4, swap_space=4)
    llm_tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens = max_tokens)
    sampling_params.stop = [llm_tokenizer.eos_token]
    return llm, sampling_params

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, default = "model_path")
    args = parser.parse_args()
    dataset = "factual_recall"
    prompt = "without_system_prompt"
    llm, sampling_params = init_llm(args.model)
    args.model = os.path.split(args.model)[1]
    data_all = {}
    for task in [cities, politicians, artists, landmarks, festivals, sports_persons]:
        for lang in LANGUAGES:
            task_data = task[lang]
            for i in range(len(task_data)):
                for j in range(len(LANGUAGES)):
                    if LANGUAGES[j] not in data_all:
                        data_all[LANGUAGES[j]] = []

                    if task == cities or task == landmarks:
                        ques = QUESTIONS_1[LANGUAGES[j]].format(task_data[i][j])
                        label = ("cities" if task == cities else "lakes")
                    elif task == festivals:
                        ques = QUESTIONS_3[LANGUAGES[j]].format(task_data[i][j])
                        label = ("cities" if task == cities else "lakes")
                    else:
                        ques = QUESTIONS_2[LANGUAGES[j]].format(task_data[i][j])
                        label = ("politicians" if task == politicians else "artists" if task == artists else "athletes")
                    data_all[LANGUAGES[j]].append({"question": ques, "answers": COUNTRIES[LANGUAGES[j]][lang], "label": label})

    for lang in LANGUAGES:
        data = data_all[lang]


        if prompt == "without_system_prompt":
            sys = ""
        prompts = []
        for i in range(len(data)):
            prompts.append(data[i]["question"])
        llm_outputs = get_llm_outputs(llm, prompts, sampling_params, sys)


        if not os.path.exists(f"generations/{dataset}/{args.model}/{prompt}/"):
            os.makedirs(f"generations/{dataset}/{args.model}/{prompt}/")
        with open(f"generations/{dataset}/{args.model}/{prompt}/{lang}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "answers", "label", "llm_output"])
            for i in range(len(data)):
                writer.writerow([data[i]["question"], data[i]["answers"], data[i]["label"], llm_outputs[i]])
            

    





