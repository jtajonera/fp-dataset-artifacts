import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json

import textattack
from textattack.augmentation import EmbeddingAugmenter, CharSwapAugmenter, WordNetAugmenter

# To use augmented data run the following command: dataset = datasets.Dataset.from_csv("aug_data.csv").remove_columns('Unnamed: 0')

# To see effect of adversarial training run: textattack attack --recipe textfooler --model ./trained_model/ --num-examples 100 --dataset-from-huggingface snli

# Dataset Size (multiple of 3)
DATASET_SIZE = 500000

# Function to augment data
def augmentData(augmenter, train_aug, start, end):
    # train_aug['premise'][start:end] = [augmenter.augment(train_aug['premise'][x])[0] for x in range(start, end)]
    # train_aug['hypothesis'][start:end] = [augmenter.augment(train_aug['hypothesis'][x])[0] for x in range(start, end)]  

    for x in range(start, end):
        train_aug['premise'][x] = augmenter.augment(train_aug['premise'][x])[0]
        train_aug['hypothesis'][x] = augmenter.augment(train_aug['hypothesis'][x])[0]


def main():

    # Dataset selection
    dataset_id = 'snli'
    eval_split = 'validation'
    dataset = datasets.load_dataset("snli")
    
    # Grab amount of data we would like to train
    train_aug = dataset['train']

    print(train_aug)
    train_aug = train_aug[0:DATASET_SIZE]

    # Initialize Augmenters
    eAugmenter = EmbeddingAugmenter()
    # cAugmenter = CharSwapAugmenter()
    wAugmenter = WordNetAugmenter()

    step = int(DATASET_SIZE / 2)

    # Augment data
    augmentData(eAugmenter, train_aug, 0, step)
    print("Completed 1/2")
    # augmentData(cAugmenter, train_aug, step, step * 2)
    # print("Completed 2/2")
    augmentData(wAugmenter, train_aug, step, step * 2)
    print("Completed 2/2")

    # Save data to CSV
    test = datasets.Dataset.from_dict(train_aug) 
    test.to_csv("aug_data.csv")

if __name__ == "__main__":
    main()
