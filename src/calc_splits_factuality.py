import random
import csv
import os
from argparse import ArgumentParser
from utils import SYSTEM_PROMPT_OURS, PROMPT_OURS


LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

random.seed(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, default = "model_path")
    args = parser.parse_args()
    args.model = os.path.split(args.model)[1]
    dataset = "factual_recall"
    countries = []
    prompt = "without_system_prompt"
    data_scores = {prompt: []}
    for lang in LANGUAGES:
        data = pd.read_csv(f"results/{dataset}/{args.model}/{args.prompt}/{lang}.csv")
        data1 = pd.read_csv(f"results/{dataset}/{args.model}/{args.prompt}/English.csv")
        results = {}
        for i in range(len(data)):
            llm_outputs = data.iloc[i]["llm_output"]
            print(llm_outputs, lang, args.model, dataset)
            if data1.iloc[i]["answers"] not in results:
                results[data1.iloc[i]["answers"]] = 0
            if "[4]" in llm_outputs:
                results[data1.iloc[i]["answers"]] += 1


        print(f"Accuracy for {lang}: {results}")
        with open(f"results/{dataset}/{args.model}/splits_{prompt}.csv", "a", errors = "ignore", encoding = "utf-8") as f:
            writer = csv.writer(f)
            if LANGUAGES.index(lang) == 0:
                writer.writerow(["LANGUAGES"] + list(results.keys()))
                countries = results.keys()
            writer.writerow([lang] + [results[country] for country in countries])


    





