import os

import numpy as np
import pandas as pd
import torch
import random
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

from preprocessing.main import Preprocessing
from utils import split_dataset, create_dataloader

# checks if the GPU is avalible for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config:
    """Configuration class to hold all settings and hyperparameters"""

    seed_value = 2042
    epochs = 4
    lr = 2e-5
    batch_size = 32
    model_name = "bert-base-uncased"
    num_labels = 6
    data_file = "cyberbullying_tweets.csv"
    model_path = "bert_finetuned_model.pth"

    @staticmethod
    def set_seed():
        random.seed(Config.seed_value)
        np.random.seed(Config.seed_value)
        torch.manual_seed(Config.seed_value)
        torch.cuda.manual_seed_all(Config.seed_value)


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
        input_ids = batch[0].to(device)  # move input data to the "device"
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

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
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

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

    sentiments = [
        "religion",
        "age",
        "ethnicity",
        "gender",
        "not_cyberbullying",
        "other_cyberbullying",
    ]
    # generate detailed perfomance report
    report = classification_report(all_labels, all_labels, target_names=sentiments)
    conf_matrix = confusion_matrix(
        all_labels, all_labels, " BERT Sentiment Analysis\nConfusion Matrix", sentiments
    )
    return (
        total_loss / len(data_loader),
        correct_predictions.double() / len(data_loader.dataset),
        report,
        conf_matrix,
    )


def main():
    Config.set_seed()

    # read in the data and save as a dataframe
    df = read_data(Config.data_file)

    # do initial preprocessing
    preprocessing = Preprocessing()
    preprocessing.clean_tweets(df)
    preprocessing.code_sentiment(df)

    # split the dataset into training, testing and validation sets
    X_train, X_test, y_train, y_test = split_dataset(
        df, Config.seed_value, df["text_clean"].values, df["cyberbullying_type"].values
    )
    X_train, X_valid, y_train, y_valid = split_dataset(
        df, Config.seed_value, X_train, y_train
    )

    ros = RandomOverSampler()
    X_train_os, y_train_os = ros.fit_resample(
        np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1)
    )
    X_train_os = X_train_os.flatten()
    y_train_os = y_train_os.flatten()

    # tokenize the inputs
    train_inputs, train_masks = preprocessing.tokenizer(X_train_os)
    val_inputs, val_masks = preprocessing.tokenizer(X_valid)
    test_inputs, test_masks = preprocessing.tokenizer(X_test)

    # convert to tensors
    train_labels, val_labels, test_labels = preprocessing.convert_to_tensors(
        y_train_os, y_valid, y_test
    )

    # get and initialize the loaders
    train_loader = create_dataloader(
        train_inputs, train_masks, train_labels, "random", batch_size=Config.batch_size
    )
    val_loader = create_dataloader(
        val_inputs, val_masks, val_labels, "sequential", batch_size=Config.batch_size
    )
    test_loader = create_dataloader(
        test_inputs, test_masks, test_labels, "sequential", batch_size=Config.batch_size
    )

    # load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        Config.model_name, num_labels=Config.num_labels
    )

    model = model.to(device)

    # initialize the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=Config.lr, correct_bias=False)

    # fine tuning training loop
    print("Starting fine-tuning process...")
    for epoch in range(Config.epochs):
        print(f"Epoch {epoch + 1}/{Config.epochs}")

        # training
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")

        # validation
        val_loss, val_acc, report, conf_matrix = eval_model(model, val_loader, device)

        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        print("Classification Report:\n", report)
        print(conf_matrix)

    # save the model
    torch.save(model.state_dict(), Config.model_path)
    print(f"Model saved to {Config.model_path}")

    # evaluate on test set
    print("Evaluating on the test set...")
    test_loss, test_acc, test_report, test_conf_matrix = eval_model(
        model, test_loader, device
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    print("Test Classification Report:\n", test_report)
    print(test_conf_matrix)


if __name__ == "__main__":
    main()
