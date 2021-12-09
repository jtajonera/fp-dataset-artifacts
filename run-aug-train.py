import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

import textattack

# Run the following command: python3 run-aug-train.py --model ./trained_model/ --output_dir ./outputs/ --dataset snli

def main():
    argp = HfArgumentParser(TrainingArguments)

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
                      
    training_args, args = argp.parse_args_into_dataclasses()
    # Dataset selection
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    if args.dataset.endswith('.csv'):
        dataset_id = None
        print(dataset)
        dataset = datasets.Dataset.from_csv(args.dataset).remove_columns('Unnamed: 0') 
        # raise Exception()
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} 

    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = dataset['train']
    eval_dataset = dataset[eval_split]

    training_args = textattack.TrainingArgs(
        num_epochs=3,
        num_clean_epochs=0,
        num_train_adv_examples=1000,
        learning_rate=0.00001,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        log_to_tb=True,
    )   

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)
    
    format_data_train = [((train_dataset[x]['premise'], train_dataset[x]['hypothesis']), train_dataset[x]['label']) for x in range(len(train_dataset))]#len(train_dataset))
    format_data_eval = [((eval_dataset[x]['premise'], eval_dataset[x]['hypothesis']), eval_dataset[x]['label']) for x in range(len(eval_dataset))]#len(train_dataset))

    # print(format_data_train)
    tData = textattack.datasets.Dataset(format_data_train, input_columns=("premise", "hypothesis"))
    eData = textattack.datasets.Dataset(format_data_eval, input_columns=("premise", "hypothesis"))
    attack = textattack.attack_recipes.TextFoolerJin2019.build(model_wrapper)

    trainer = textattack.Trainer(
        model_wrapper,
        "classification",
        attack,
        tData,
        eData,
        training_args
    )
    trainer.train()


if __name__ == "__main__":
    main()
