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

train_dataset = textattack.datasets.HuggingFaceDataset("snli", split="train")
eval_dataset = textattack.datasets.HuggingFaceDataset("snli", split="test")

# Code Inspired by: https://textattack.readthedocs.io/en/latest/api/trainer.html#trainer

# Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
training_args = textattack.TrainingArgs(
    num_epochs=3,
    num_clean_epochs=1,
    num_train_adv_examples=1000,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification",
    attack,
    train_dataset,
    eval_dataset,
    training_args
)

trainer.train()

