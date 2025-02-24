
import random
import os
import pandas as pd
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from scipy import stats


LANGUAGES = [
    "English", "Hindi", "Chinese", "Russian", "Arabic", "French", "Nepali", 
    "Japanese", "Ukrainian", "Greek", "Turkish", "Swahili", "Thai"
]

HIGH = ["English", "Chinese", "French", "Japanese"]
MEDIUM = ["Hindi", "Russian", "Arabic", "Greek", "Turkish"]
LOW = ["Nepali", "Ukrainian", "Swahili", "Thai"]

COUNTRIES = [
    "United States",
    "India",
    "China",
    "Russia",
    "Saudi Arabia",
    "France",
    "Nepal",
    "Japan",
    "Ukraine",
    "Greece",
    "Turkey",
    "Kenya",
    "Thailand"
]

random.seed(0)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type = str, default = "Hindi")
    args = parser.parse_args()
    args.model = os.path.split(args.model)[1]
    dataset = "factual_recall"
    prompt = "without_system_prompt"
    countries = []
    data_scores = {}
    for lang in LANGUAGES:
        data = pd.read_csv(f"results/{dataset}/{args.model}/{prompt}/{lang}.csv")
        data1 = pd.read_csv(f"results/{dataset}/{args.model}/{prompt}/English.csv")
        results = {}
        count = {}
        for i in range(len(data)):
            llm_outputs = data.iloc[i]["llm_output"]
            if data1.iloc[i]["answers"] not in count:
                count[data1.iloc[i]["answers"]] = 0
            count[data1.iloc[i]["answers"]] += 1
            if data1.iloc[i]["answers"] not in results:
                results[data1.iloc[i]["answers"]] = 0
            if "[4]" in llm_outputs:
                results[data1.iloc[i]["answers"]] += 1
        for key in results.keys():
            results[key] = round((results[key]/count[key]), 2)
        data_scores[lang] = results
    
    data1 = []
    data2 = []
    countries = results.keys()
    for lang in LANGUAGES:
        for country in COUNTRIES:
            if LANGUAGES.index(lang) == COUNTRIES.index(country):
                data2.append(data_scores[lang][country])
            else:
                data1.append(data_scores[lang][country])
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    mean1 = np.mean(data1)
    var1 = np.std(data1)
    mean2 = np.mean(data2)
    var2 = np.std(data2)
    t_stat, p_value = stats.ttest_ind(data1, data2)
    frs = 1.5 * ((1 / (mean1 + mean2 + 1)) - (1 / 3))
    kts = 2 * ((1 / (abs(mean1 - mean2) + 1)) - (1 / 2))
    x_fakt = (2 * frs * kts) / (frs + kts)
    
print(f"Factual Recall Score: {frs}")
print(f"Knowledge Transfer Score: {kts}")
print(f"Cross-Lingual Factual Knowledge Transferability Score: {x_fakt}")


    





