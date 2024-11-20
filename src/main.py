from transformers import pipeline

# Load the sentiment-analysis pipeline directly (fine-tuned model for sentiment)
classifier = pipeline("sentiment-analysis")

# Input text
text = "ok boomer"

# Get prediction
result = classifier(text)
print(result)  # Outputs the label (positive/negative) and confidence score
