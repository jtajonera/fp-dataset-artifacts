import torch
import textattack
from transformers import AutoTokenizer, AutoConfig
from textattack.models.wrappers import ModelWrapper

import transformers

MODEL_LOC = "./trained_model/"

modelConfig = AutoConfig.from_pretrained(MODEL_LOC)

model = transformers.ElectraForSequenceClassification(modelConfig)

tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC, use_fast=True)

model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

print("Loaded Model", type(model))

# Attack Recipes Available: https://textattack.readthedocs.io/en/latest/apidoc/textattack.attack_recipes.html#attack-recipes

# Code Inspired by: https://textattack.readthedocs.io/en/latest/0_get_started/quick_api_tour.html#attacking-a-bert-model

dataset = textattack.datasets.HuggingFaceDataset("snli", split="test")
attack = textattack.attack_recipes.BAEGarg2019.build(model_wrapper)

# Attack 20 samples with CSV logging and checkpoint saved every 5 interval

attack_args = textattack.AttackArgs(num_examples=20, log_to_csv="log.csv", checkpoint_interval=5, checkpoint_dir="checkpoints", disable_stdout=True)
attacker = textattack.Attacker(attack, dataset, attack_args)
attacker.attack_dataset()

