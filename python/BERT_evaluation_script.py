import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
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


# Define a filtering function to keep only examples with a single label
def filter_single_label(example):
    return len(example["labels"]) == 1


# Apply the filter to all splits (train, validation, test)
filtered_dataset = dataset.filter(filter_single_label)

test_dataset = filtered_dataset["test"]


# Preprocess function (tokenization)
def preprocess_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=200
    )


# Tokenize the test dataset
tokenized_test = test_dataset.map(preprocess_function, batched=True)


# Custom collate function to convert dataset batches into tensors
def custom_collate_fn(batch):
    # Extract inputs and labels from the batch
    input_ids = [torch.tensor(example["input_ids"]) for example in batch]
    attention_masks = [torch.tensor(example["attention_mask"]) for example in batch]
    labels = [
        torch.tensor(example["labels"][0]) for example in batch
    ]  # Use the single label

    # Pad the sequences to the same length
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)

    # Return the padded inputs and labels
    return {"input_ids": input_ids, "attention_mask": attention_masks, "label": labels}


# DataLoader for batching the test data with the custom collate function
test_dataloader = DataLoader(
    tokenized_test, batch_size=16, collate_fn=custom_collate_fn
)


# Updated predict function (no changes needed here)
def predict(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to the appropriate device
            inputs = {
                key: value.to(device) for key, value in batch.items() if key != "label"
            }
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Get the predicted class for each example in the batch
            probs = F.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)

            # Append predictions and true labels
            predictions.extend(predicted_classes.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


# Make predictions on the test dataset
predictions, true_labels = predict(model, test_dataloader, device)

# Calculate accuracy
accuracy = np.mean(np.array(predictions) == np.array(true_labels))
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
