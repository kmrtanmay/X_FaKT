MODEL="<MODEL_PATH>"

# Perform generations for the given model
python src/factual_gen.py --model $MODEL
python src/incontext_gen.py --model $MODEL
python src/counter_factual_gen.py --model $MODEL


# Perform evaluations for the given model
python src/llm_eval.py --model $MODEL --data factual_recall
python src/llm_eval.py --model $MODEL --data incontext_recall
python src/llm_eval.py --model $MODEL --data counter_factual