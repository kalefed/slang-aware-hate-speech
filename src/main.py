from transformers import BertForSequenceClassification, BertTokenizer, pipeline
import torch

model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#  tokenize the text
text = "BERT is amazing!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

# make predictions
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits).item()
print("Predicted Sentiment Class:", predicted_class)

unmasker = pipeline("fill-mask", model="bert-base-uncased")
unmasker("Artificial Intelligence [MASK] take over the world.")
