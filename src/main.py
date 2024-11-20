from transformers import BertModel, BertTokenizer, DistilBertForSequenceClassification
import torch

num_labels = 5

# get BERT base model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=num_labels) 

classifier = nn.Linear(47986, num_labels)

model = nn.Sequential(model, classifier)

# define an optimizer and loss function for the model
# cross-entropy loss for the classifiers
criterion = nn.CrossEntropyLoss()  
optimizer = AdamW(model.parameters(), lr=2e-5)

# create DataLoader objects from the datasets to feed data into the model
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)


# Define a training function to train your model
def train(model, optimizer, train_loader, criterion):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch 
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Training loss: {total_loss/len(train_loader)}')