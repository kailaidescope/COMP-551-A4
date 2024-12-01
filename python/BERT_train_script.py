import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import sys

if len(sys.argv) == 1:
    output_path = "."
    head_name = "classification_head"
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
    head_name = "classification_head"
elif len(sys.argv) == 3:
    output_path = sys.argv[1]
    head_name = sys.argv[2]
else:
    print("Usage: python test_graph_saving.py [output_path] [head_name]")
    sys.exit(1)

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

# Freeze the base BERT model layers and only train the classification head
for param in model.bert.parameters():
    param.requires_grad = False

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

print("Fine-tuning the model on the GoEmotions dataset...")

# Step 1: Load the GoEmotions dataset
dataset = load_dataset("google-research-datasets/go_emotions")


# Define a filtering function to keep only examples with a single label
def filter_single_label(example):
    return len(example["labels"]) == 1


# Apply the filter to all splits (train, validation, test)
filtered_dataset = dataset.filter(filter_single_label)


# Step 2: Preprocess the dataset (tokenize)
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",  # Pad to a fixed length
        max_length=200,  # Set a fixed max length (can adjust as needed)
    )


# Tokenize the dataset
tokenized_datasets = filtered_dataset.map(preprocess_function, batched=True)


# Step 3: Define the compute metric function for evaluation
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}


# Step 4: Set up training arguments
# Define training arguments with minimal output
training_args = TrainingArguments(
    output_dir=f"{output_path}/results",  # Still need an output directory, but no logging or saving
    evaluation_strategy="epoch",  # Evaluate every epoch
    learning_rate=2e-5,  # Learning rate for training
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=7,  # Number of epochs
    weight_decay=0.01,  # Weight decay strength
    logging_strategy="no",  # No logging
    save_strategy="no",  # No saving
    push_to_hub=False,  # Don't push model to Hugging Face Hub
    report_to="none",  # Disable reporting to tracking tools like TensorBoard, etc.
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)


# Step 6: Train the model
trainer.train()

# Step 6.5: Save only the classification head (classifier layer)
classifier_layer = model.classifier  # This is the classification head
torch.save(classifier_layer.state_dict(), f"{output_path}/{head_name}.pth")
print("Classification head saved.")

# Step 7: Test the model again after fine-tuning
inputs = tokenizer("Hello, Hugging Face!", return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass through the fine-tuned model
with torch.no_grad():  # No need to compute gradients for inference
    outputs = model(**inputs)

# Get the logits (raw predictions) from the model output
logits = outputs.logits

# Convert logits to probabilities (using softmax)
probabilities = F.softmax(logits, dim=-1)

# Get the predicted class label (the index of the max probability)
predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

# Print the new prediction after training
print(
    f"Predicted class after fine-tuning: {label_map.get(str(predicted_class_idx), 'Unknown')}"
)
print(f"Predicted probabilities after fine-tuning: {probabilities}")
