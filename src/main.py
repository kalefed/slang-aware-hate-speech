import torch # handles model training and evaluation
from transformers import BertForSequenceClassification, AdamW # provides tools for using BERT
from sklearn.metrics import classification_report # generates a report to see how wel the model is performing based on accuracy and precision
from tqdm import tqdm # visual progress bar 

# checks if the GPU is avalible for faster processing
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training function
def train_epoch(model, data_loader, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()  # tells model we are training so it can update the weights
    total_loss = 0 # keep track of total loss in this epoch
    correct_predictions = 0 # count how many predictions model got right

    for batch in tqdm(data_loader, desc="Training"): # loops through batchs of dataa
        input_ids = batch['input_ids'].to(device) # move input data to the "device"
        attention_mask = batch['attention_mask'].to(device) 
        labels = batch['labels'].to(device)

        optimizer.zero_grad()  # clears the prev gradients
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # get the loss for this batch 
        logits = outputs.logits  # get the models predictions

        total_loss += loss.item() # add batches loss to the total loss
        loss.backward()  # backpropagation to calculate gradients  
        optimizer.step()  # update the model weights using the optimizer 

        # checks how many predictions are correct
        _, preds = torch.max(logits, dim=1) # get predicted class for each input
        correct_predictions += torch.sum(preds == labels) # compare predictions with actual labels

    # return avg loss and the accruacy 
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

# evaluation function
def eval_model(model, data_loader, device):
    """
    Evaluates the model on the validation dataset.
    """
    model.eval()  # tells model we are evaluating so weights don't get updated
    total_loss = 0 # keep track of total loss during eval
    correct_predictions = 0 # count how many predictions are correct
    all_labels = [] # store all true labels
    all_preds = [] # store all predicted labels

    with torch.no_grad(): # no need to calc gradients during evaluation
        for batch in tqdm(data_loader, desc="Evaluating"): # loop through validation data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss # get the loss for this batch
            logits = outputs.logits # get predictions

            total_loss += loss.item() # add batches loss to the total
            _, preds = torch.max(logits, dim=1) # get predicted class for each input
            correct_predictions += torch.sum(preds == labels) # compare predictions with actual labels

            # save true and predicted labels for classification report
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # generate detailed perfomance report
    report = classification_report(all_labels, all_preds, zero_division=1)
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), report

# main function
def main(train_loader, val_loader):
    """
    Fine-tunes a BERT model for text classification using preprocessed data.
    Assumes `train_loader` and `val_loader` are DataLoader objects created elsewhere.
    """
    # model and training parameters
    EPOCHS = 4 # num of times the model will go through the training data
    LR = 2e-5 # controls how much the model updates the weights

    # load the pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model = model.to(device) # move the model to the device (GPU or CPU)

    # initialize the optimizer
    optimizer = AdamW(model.parameters(), lr=LR, correct_bias=False)

    # training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Training Loss: {train_loss}, Training Accuracy: {train_acc}")

        val_loss, val_acc, report = eval_model(model, val_loader, device)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
        print("Classification Report:\n", report)

    # save the trained model
    MODEL_PATH = 'bert_finetuned_model.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Execute the main function
if __name__ == "__main__":
    # replace these with the  DataLoader objects from the preprocessing script
    train_loader = None  # Example: replace with preprocessed training DataLoader
    val_loader = None    # Example: replace with preprocessed validation DataLoader

    # check to make sure dataloader values arent none
    if train_loader is not None and val_loader is not None:
        main(train_loader, val_loader)
    else:
        print("Please provide valid DataLoader objects for training and validation.")
