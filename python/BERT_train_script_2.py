import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset, load_metric
import numpy as np
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

print("Starting BERT fine-tuning script")

learning_rate = 0.01
num_epochs = 8
batch_size = 16
weight_decay = 0.01

print(
    "Learning rate:",
    learning_rate,
    "\nNumber of epochs:",
    num_epochs,
    "\nBatch size:",
    batch_size,
    "\nWeight decay:",
    weight_decay,
)

# Path to the local directory containing the saved model
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"

model_path = bert_path

print("Loading model")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=28
)  # 27 emotions + neutral

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Move the model to the selected device (either GPU or CPU)
model.to(device)

only_train_head = True

if only_train_head:
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        print("Name:", name, " - Size:", param.size())

    # Only the classification head will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True

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

print("Fine-tuning the model on the GoEmotions dataset...")

# Step 1: Load the GoEmotions dataset
dataset = load_dataset("google-research-datasets/go_emotions")


# Define a filtering function to keep only examples with a single label
def filter_single_label(example):
    return len(example["labels"]) == 1


# Apply the filter to all splits (train, validation, test)
filtered_dataset = dataset.filter(filter_single_label)

print("Filtered dataset:\n", filtered_dataset)


# Step 2: Preprocess the dataset (tokenize)
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Tokenize the dataset
tokenized_datasets = filtered_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer)


# Step 3: Define the compute metric function for evaluation
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
    }


# Step 4: Set up training arguments
# Define training arguments with minimal output
training_args = TrainingArguments(
    output_dir=f"{output_path}/results",  # Still need an output directory, but no logging or saving
    evaluation_strategy="epoch",  # Evaluate every epoch
    learning_rate=learning_rate,  # Learning rate for training
    per_device_train_batch_size=batch_size,  # Batch size for training
    per_device_eval_batch_size=batch_size,  # Batch size for evaluation
    num_train_epochs=num_epochs,  # Number of epochs
    weight_decay=weight_decay,  # Weight decay strength
    logging_strategy="epoch",  # No logging
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
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# Step 6: Train the model
trainer.train()

# Step 7: Save only the classification head (classifier layer)
classifier_layer = model.classifier  # This is the classification head
torch.save(classifier_layer.state_dict(), f"{output_path}/{head_name}.pth")
print("Classification head saved.")

# Step 8: Test the model

predictions = trainer.predict(tokenized_datasets["test"])
class_predictions = np.argmax(predictions.predictions, axis=1)
print(
    "Predictions shape: ",
    predictions.predictions.shape,
    "\nLabels shape: ",
    predictions.label_ids.shape,
    "\nClass Predictions: ",
    class_predictions,
)
print(
    "Metrics:\nF1:",
    f1_score(predictions.label_ids, class_predictions, average="macro"),
    "\nAccuracy:",
    accuracy_score(predictions.label_ids, class_predictions),
)
