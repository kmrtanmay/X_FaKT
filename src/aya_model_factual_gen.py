import random
import csv
import os
from argparse import ArgumentParser
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import cities, sports_persons, landmarks, politicians, festivals, artists, QUESTIONS_1, QUESTIONS_2, QUESTIONS_3, COUNTRIES
import torch
from tqdm import tqdm

random.seed(0)

LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

def init_aya_model(checkpoint="CohereForAI/aya-101"):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map='auto')
    return tokenizer, model

def batched_inference(tokenizer, model, prompts, batch_size=16, max_new_tokens=128):
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        outputs.extend(decoded_outputs)
    return outputs

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="CohereForAI/aya-101")
    args = parser.parse_args()

    dataset = "factual_recall"
    prompt = "without_system_prompt"

    tokenizer, aya_model = init_aya_model(args.model)

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
                        label = "cities" if task == cities else "lakes"
                    elif task == festivals:
                        ques = QUESTIONS_3[LANGUAGES[j]].format(task_data[i][j])
                        label = "cities" if task == cities else "lakes"
                    else:
                        ques = QUESTIONS_2[LANGUAGES[j]].format(task_data[i][j])
                        label = "politicians" if task == politicians else "artists" if task == artists else "athletes"

                    data_all[LANGUAGES[j]].append({
                        "question": ques,
                        "answers": COUNTRIES[LANGUAGES[j]][lang],
                        "label": label
                    })

    for lang in LANGUAGES:
        data = data_all[lang]

        prompts = [item["question"] for item in data]

        llm_outputs = batched_inference(tokenizer, aya_model, prompts)

        output_dir = f"generations/{dataset}/{args.model}/{prompt}/"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{output_dir}/{lang}.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Question", "answers", "label", "llm_output"])
            for i in range(len(data)):
                writer.writerow([data[i]["question"], data[i]["answers"], data[i]["label"], llm_outputs[i]])
