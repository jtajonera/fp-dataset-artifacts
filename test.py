import textattack
import transformers

import datasets

from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy

# NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
task_kwargs = {'num_labels': 3} 

model = transformers.AutoModelForSequenceClassification.from_pretrained("google/electra-small-discriminator",  **task_kwargs)
tokenizer = transformers.AutoTokenizer.from_pretrained("google/electra-small-discriminator", use_fast=True)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# We only use DeepWordBugGao2018 to demonstration purposes.
attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)

train_data = textattack.datasets.HuggingFaceDataset("snli", split="train")
eval_data = textattack.datasets.HuggingFaceDataset("snli", split="test")

# # Preparing Dataset
# prepare_train_dataset = prepare_eval_dataset = \
#             lambda exs: prepare_dataset_nli(exs, tokenizer, 128)

# dataset = datasets.load_dataset("snli")
# dataset = dataset.filter(lambda ex: ex['label'] != -1)

# train_dataset = dataset['train']
# eval_dataset = dataset['validation']
# train_dataset_featurized = None
# eval_dataset_featurized = None


# train_dataset_featurized = train_dataset.map(
#             prepare_train_dataset,
#             remove_columns=train_dataset.column_names
#         )

# print(train_dataset)
# print(train_dataset.column_names)
# print(train_dataset_featurized.column_names)
# print(prepare_train_dataset)


# eval_dataset_featurized = eval_dataset.map(
#             prepare_eval_dataset,
#             remove_columns=eval_dataset.column_names
#         )

# taDataset = textattack.datasets.Dataset(train_dataset, input_columns=['premise', 'hypothesis'])

# taEval = textattack.datasets.Dataset(eval_dataset, input_columns=['premise', 'hypothesis'])
# 
# train_dataset = textattack.datasets.HuggingFaceDataset("snli", split="train")
# eval_dataset = textattack.datasets.HuggingFaceDataset("snli", split="test")

# Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
training_args = textattack.TrainingArgs(
    num_epochs=1,
    num_clean_epochs=0,
    num_train_adv_examples=1,
    learning_rate=0.001,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification",
    attack,
    train_data,
    eval_data,
    training_args
)
trainer.train()

# import transformers
# import textattack
# from textattack.shared import utils
# model_path = "./trained_model/"

# model = transformers.AutoModelForSequenceClassification.from_pretrained("./trained_model/")
# tokenizer = transformers.AutoTokenizer.from_pretrained("./trained_model/")
# model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
# dataset = textattack.datasets.HuggingFaceDataset(
#   "snli", shuffle = False
# )