import textattack
import transformers

model = transformers.AutoModelForSequenceClassification.from_pretrained("./trained_model/")
tokenizer = transformers.AutoTokenizer.from_pretrained("./trained_model/")
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

# We only use DeepWordBugGao2018 to demonstration purposes.
attack = textattack.attack_recipes.DeepWordBugGao2018.build(model_wrapper)
train_dataset = textattack.datasets.HuggingFaceDataset("snli", split="train")
eval_dataset = textattack.datasets.HuggingFaceDataset("snli", split="test")

# Train for 3 epochs with 1 initial clean epochs, 1000 adversarial examples per epoch, learning rate of 5e-5, and effective batch size of 32 (8x4).
training_args = textattack.TrainingArgs(
    num_epochs=1,
    num_clean_epochs=0,
    num_train_adv_examples=1,
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