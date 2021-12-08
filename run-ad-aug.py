import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

import textattack
from textattack.augmentation import EmbeddingAugmenter

NUM_PREPROCESSING_WORKERS = 2

# To use augmented data run the following command: dataset = datasets.Dataset.from_csv("aug_data.csv").remove_columns('Unnamed: 0')

def main():
    # Dataset selection

    dataset_id = 'snli'
    eval_split = 'validation'
    dataset = datasets.load_dataset("snli")

    train_aug = dataset['train']

    # Augments the first 20,000 cases in train
    train_aug = train_aug[0:20000]
    
    augmenter = EmbeddingAugmenter()

    train_aug['premise'] = [augmenter.augment(x)[0] for x in train_aug['premise']]
    train_aug['hypothesis'] = [augmenter.augment(x)[0] for x in train_aug['hypothesis']]

    print(type(train_aug))

    test = datasets.Dataset.from_dict(train_aug) 

    test.to_csv("aug_data.csv")

if __name__ == "__main__":
    main()
