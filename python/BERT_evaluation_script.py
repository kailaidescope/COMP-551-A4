import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForSequenceClassification,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import sys

if len(sys.argv) == 1:
    input_head_path = "."
elif len(sys.argv) == 2:
    input_head_path = sys.argv[1]
else:
    print("Usage: python test_graph_saving.py [input_path]")
    sys.exit(1)

# Load the tokenizer and base BERT model
bert_path = "/opt/models/bert-base-uncased"  # Path to your BERT model
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# Initialize the base model (without the classification head)
model = AutoModelForSequenceClassification.from_pretrained(bert_path, num_labels=28)

# Load the classification head from the specified path
classification_head_path = input_head_path  # Path to the saved classification head
classification_head = torch.load(classification_head_path)

# Replace the model's classification head with the loaded one
model.classifier.load_state_dict(classification_head)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the GoEmotions dataset
dataset = load_dataset("google-research-datasets/go_emotions")
test_dataset = dataset["test"]


# Preprocess function (tokenization)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)


# Tokenize the test dataset
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# DataCollator for dynamic padding
data_collator = DataCollatorForSequenceClassification(tokenizer, padding="longest")

# DataLoader for batching the test data
test_dataloader = DataLoader(tokenized_test, batch_size=16, collate_fn=data_collator)


# Function to make predictions on the test dataset
def predict(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to the appropriate device
            inputs = {key: value.to(device) for key, value in batch.items()}

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Get the predicted class for each example in the batch
            probs = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)

            # Append predictions and true labels
            predictions.extend(predicted_classes.cpu().numpy())
            true_labels.extend(
                batch["label"]
            )  # Assuming the labels are in the "label" column

    return predictions, true_labels


# Make predictions on the test dataset
predictions, true_labels = predict(model, test_dataloader, device)

# Calculate accuracy
accuracy = (predictions == true_labels).mean()
print(f"Test Accuracy: {accuracy:.4f}")

# Optionally, map the predictions to the labels (for example, emotions)
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

# Print some sample predictions
print("Sample predictions:")
for i in range(10):
    print(f"Text: {test_dataset[i]['text']}")
    print(
        f"True Label: {label_map[str(true_labels[i])]} | Predicted: {label_map.get(str(predictions[i]), 'Unknown')}"
    )
