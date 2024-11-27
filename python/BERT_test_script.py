import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

print("Starting BERT test script")

# Path to the local directory containing the saved model
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"

model_path = bert_path

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=28
)  # 27 emotions + neutral

# Test the model with a sample input
input_text = "Hello, Hugging Face!"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Move the model to the selected device (either GPU or CPU)
model.to(device)

# Move input tensors to GPU if available
inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass through the model
with torch.no_grad():  # No need to compute gradients for inference
    outputs = model(**inputs)

# Get the logits (raw predictions) from the model output
logits = outputs.logits

# Convert logits to probabilities (using softmax)
probabilities = F.softmax(logits, dim=-1)

# Get the predicted class label (the index of the max probability)
predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

# If you have a label map (e.g., emotions or sentiments), you can map the index to the label
label_map = {
    "0": "admiration",
    "1": "amusement",
    "2": "anger",
    "3": "annoyance",
    "4": "approval",
    "5": "caring",
    "6": "confusion",
    "7": "curiosity",
    "8": "desire",
    "9": "disappointment",
    "10": "disapproval",
    "11": "disgust",
    "12": "embarrassment",
    "13": "excitement",
    "14": "fear",
    "15": "gratitude",
    "16": "grief",
    "17": "joy",
    "18": "love",
    "19": "nervousness",
    "20": "optimism",
    "21": "pride",
    "22": "realization",
    "23": "relief",
    "24": "remorse",
    "25": "sadness",
    "26": "surprise",
    "27": "neutral",
}

# Print the result
print(f"Predicted class: {label_map.get(str(predicted_class_idx), 'Unknown')}")
print(f"Predicted probabilities: {probabilities}")
