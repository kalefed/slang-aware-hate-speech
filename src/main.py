import os

import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

from preprocessing.main import Preprocessing
from utils import split_dataset, create_dataloader


# checks if the GPU is avalible for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set up seed value
seed_value = 2042
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)


def read_data(file_name):
    """Reads in the data from the given csv file as a dataframe.

    Args:
        file_name (string): the input file name

    Returns:
        dataframe: input data as a dataframe
    """
    file_dir = os.path.dirname(__file__)
    csv_path = os.path.join(file_dir, "..", "data", file_name)
    return pd.read_csv(csv_path)


# training function
def train_epoch(model, data_loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()  # tells model we are training so it can update the weights
    total_loss = 0  # keep track of total loss in this epoch
    correct_predictions = 0  # count how many predictions model got right

    for batch in tqdm(data_loader, desc="Training"):  # loops through batchs of dataa
        input_ids = batch["input_ids"].to(device)  # move input data to the "device"
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()  # clears the prev gradients
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        loss = outputs.loss  # get the loss for this batch
        logits = outputs.logits  # get the models predictions

        total_loss += loss.item()  # add batches loss to the total loss
        loss.backward()  # backpropagation to calculate gradients
        optimizer.step()  # update the model weights using the optimizer

        # checks how many predictions are correct
        _, preds = torch.max(logits, dim=1)  # get predicted class for each input
        correct_predictions += torch.sum(
            preds == labels
        )  # compare predictions with actual labels

    # return avg loss and the accruacy
    return total_loss / len(data_loader), correct_predictions.double() / len(
        data_loader.dataset
    )


# evaluation function
def eval_model(model, data_loader, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()  # tells model we are evaluating so weights don't get updated
    total_loss = 0  # keep track of total loss during eval
    correct_predictions = 0  # count how many predictions are correct
    all_labels = []  # store all true labels
    all_preds = []  # store all predicted labels

    with torch.no_grad():  # no need to calc gradients during evaluation
        for batch in tqdm(
            data_loader, desc="Evaluating"
        ):  # loop through validation data
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss  # get the loss for this batch
            logits = outputs.logits  # get predictions

            total_loss += loss.item()  # add batches loss to the total
            _, preds = torch.max(logits, dim=1)  # get predicted class for each input
            correct_predictions += torch.sum(
                preds == labels
            )  # compare predictions with actual labels

            # save true and predicted labels for classification report
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # generate detailed perfomance report
    report = classification_report(all_labels, all_preds, zero_division=1)
    return (
        total_loss / len(data_loader),
        correct_predictions.double() / len(data_loader.dataset),
        report,
    )


def main():
    # read in the data and save as a dataframe
    df = read_data("testtweets.csv")

    # do initial preprocessing
    preprocessing = Preprocessing()
    preprocessing.clean_tweets(df)
    preprocessing.code_sentiment(df)

    # split the dataset into training, testing and validation sets
    # TODO - there is an error with this: The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2
    # NOTE - the error is because the classes are unbalanced. there is some code in the notebook which fixes this so it should be integrated
    X_train, X_test, y_train, y_test = split_dataset(
        df, seed_value, df["text_clean"], df["cyberbullying_type"]
    )
    X_train, X_valid, y_train, y_valid = split_dataset(df, seed_value, X_train, y_train)

    ros = RandomOverSampler()
    X_train_os, y_train_os = ros.fit_resample(np.array(X_train), np.array(y_train))

    # tokenize the inpits
    train_inputs, train_masks = preprocessing.tokenizer(X_train_os)
    val_inputs, val_masks = preprocessing.tokenizer(X_valid)
    test_inputs, test_masks = preprocessing.tokenizer(X_test)

    train_labels, val_labels, test_labels = preprocessing.convert_to_tensors(
        y_train_os, y_valid, y_test
    )

    # get and initialize the loaders
    train_loader = create_dataloader(
        train_inputs, train_masks, train_labels, "random", batch_size=32
    )
    val_loader = create_dataloader(
        val_inputs, val_masks, val_labels, "sequential", batch_size=32
    )
    test_loader = create_dataloader(
        test_inputs, test_masks, test_labels, "sequential", batch_size=32
    )

    # TODO - add in previous main function logic here or whatever comes next (tbd)


if __name__ == "__main__":
    main()
