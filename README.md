# Large Language Models’ Factuality Depends on the Language of Inquiry

This repo contains the code for our Benchmark.

## Directory Structure
- `generations` folder contains the generations for the models we mentioned in the paper on our dataset.
- `results` folder contains LLM evaluation results for all the models.
- `src` folder contains the scripts for generations and evaluations on our dataset.

## Setup and Usage Instructions for our Benchmark

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AggarwalTushar/multilingual_benchmark.git
   cd multilingual_benchmark

2. **Install dependencies:**
   
    Use env.yml to create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate eval
   
3. **Evalute your model on the benchmark**
   Add your model path in the eval.sh file. Use the following command to run the file:
   ```bash
   bash eval.sh
   ```
   The output will be saved in the results/<dataset_name>/<model_name> folder

4. **Calculate the incorrect numbers for each country in a language for factual-recall dataset**
   ```bash
   python src/calc_splits_factuality.py --model_path <model_name>
   ```
   The output will be saved in the results/factual_recall/<model_name> folder

5. **Get FRS, KTS and X-FAKT scores for the model**
   ```bash
   python src/eval.py --model_path <model_name>
   ```

## Model Performance Results

| Model | Associative | Non-associative | t-stat | p-value | FRS | KTS | X-FAKT |
|-------|-------------|-----------------|--------|----------|-----|-----|---------|
| Llama-3-70B | 2.36 ± 5.12 | 9.85 ± 10.54 | 2.52 | 0.01 | 0.835 | 0.862 | 0.848 |
| Gemma-2-27B | 4.23 ± 8.49 | 16.46 ± 17.07 | 2.54 | 0.01 | 0.742 | 0.783 | 0.762 |
| Phi-4 | 12.87 ± 16.51 | 30.15 ± 25.92 | 2.35 | 0.02 | 0.548 | 0.706 | 0.617 |
| Phi-3-medium-128k-Inst | 25.09 ± 29.84 | 55.57 ± 36.24 | 2.93 | <0.01 | 0.330 | 0.535 | 0.408 |
| Gemma-2-9B | 4.98 ± 6.09 | 22.32 ± 21.37 | 2.90 | <0.01 | 0.677 | 0.705 | 0.691 |
| Llama-3-8B | 4.60 ± 7.54 | 25.77 ± 19.61 | 3.85 | <0.01 | 0.649 | 0.651 | 0.650 |
| Orca-2-7B | 31.95 ± 31.65 | 56.77 ± 32.99 | 2.60 | 0.01 | 0.295 | 0.603 | 0.396 |
| Deepseek-7B-chat | 31.49 ± 30.68 | 63.73 ± 36.29 | 3.09 | <0.01 | 0.268 | 0.514 | 0.353 |
| Mistral-7B-v0.2 | 16.96 ± 15.65 | 45.25 ± 29.34 | 3.42 | <0.01 | 0.424 | 0.559 | 0.483 |
| Phi-3.5-mini | 41.85 ± 31.62 | 69.87 ± 31.23 | 3.09 | <0.01 | 0.208 | 0.563 | 0.304 |
| Phi-3-mini-128k | 42.45 ± 30.99 | 77.95 ± 33.72 | 3.65 | <0.01 | 0.181 | 0.477 | 0.262 |
| Llama-3.2-3B | 24.10 ± 17.80 | 47.48 ± 26.80 | 3.07 | <0.01 | 0.375 | 0.620 | 0.467 |
| Gemma-2-2B | 9.97 ± 14.78 | 45.77 ± 31.30 | 4.06 | <0.01 | 0.463 | 0.473 | 0.468 |
| Llama-3.2-1B | 34.74 ± 22.32 | 65.96 ± 26.98 | 4.03 | <0.01 | 0.247 | 0.524 | 0.336 |

Results show model performance based on:
- Associative and Unassociative errors (mean ± standard deviation)
- Factual Recall Score (FRS)
- Knowledge Transferability Score (KTS)
- Cross-Lingual Factual Knowledge Transferability Score (X-FAKT)

# Contact
- [Tushar Aggarwal](mailto:tushar.aggarwal53@gmail.com)
- [Tanmay Kumar](mailto:kr.tanmay147@gmail.com)
- [Ayush Agrawal](mailto:ayush.agrawal@mila.quebec)