import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
import numpy as np
import sys
import time
import matplotlib.pyplot as plt

if len(sys.argv) == 1:
    output_path = "."
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
else:
    print("Usage: python over_epochs.py [output_path]")
    sys.exit(1)

print("Starting BERT epoch experiments script")

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

# Path to the local directory containing the saved model
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"

model_path = bert_path

num_epochs = 10
results = {
    "f1": [],
    "accuracy": [],
    "duration": [],
}
start_time = time.time()
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

only_train_head = False

if only_train_head:
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    """ for name, param in model.named_parameters():
        print("Name:", name, " - Size:", param.size()) """

    # Only the classification head will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True

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
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    results["f1"].append(f1)
    results["accuracy"].append(accuracy)
    results["duration"].append(time.time() - start_time)
    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# Step 4: Set up training arguments
# Define training arguments with minimal output
training_args = TrainingArguments(
    output_dir=f"{output_path}/results",  # Still need an output directory, but no logging or saving
    evaluation_strategy="epoch",  # Evaluate every epoch
    learning_rate=2e-5,  # Learning rate for training
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=num_epochs,  # Number of epochs
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
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Step 6: Train the model
trainer.train()

# Step 7: Test the model
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

f1 = f1_score(predictions.label_ids, class_predictions, average="macro")
accuracy = accuracy_score(predictions.label_ids, class_predictions)
print(
    "Metrics:\nF1:",
    f1,
    "\nAccuracy:",
    accuracy,
)

duration = time.time() - start_time
results["final"] = {}
results["final"]["f1"] = f1
results["final"]["accuracy"] = accuracy
results["final"]["duration"] = duration

print("Results:", results)

epoch_list = range(num_epochs + 1)
f1s = results["f1"]
accuracies = results["accuracy"]
durations = results["duration"]

print("F1 scores:", f1s)
print("Accuracies:", accuracies)
print("Durations:", durations)

# Graph the results

plt.plot(epoch_list, f1s, label="F1 Score")
if only_train_head:
    plt.title("F1 Score Over Epochs for Head Fine-Tuning")
else:
    plt.title("F1 Score Over Epochs for Full Model Fine-Tuning")
plt.savefig(f"{output_path}/f1.png", dpi=300)

plt.plot(epoch_list, accuracies, label="Accuracy")
if only_train_head:
    plt.title("Accuracy Over Epochs for Head Fine-Tuning")
else:
    plt.title("Accuracy Over Epochs for Full Model Fine-Tuning")
plt.savefig(f"{output_path}/accuracy.png", dpi=300)

plt.plot(epoch_list, durations, label="Duration")
if only_train_head:
    plt.title("Duration Over Epochs for Head Fine-Tuning")
else:
    plt.title("Duration Over Epochs for Full Model Fine-Tuning")
plt.savefig(f"{output_path}/duration.png", dpi=300)

plt.plot(epoch_list, f1s, label="F1 Score")
plt.plot(epoch_list, accuracies, label="Accuracy")
if only_train_head:
    plt.title("F1 Score and Accuracy Over Epochs for Head Fine-Tuning")
else:
    plt.title("F1 Score and Accuracy Over Epochs for Full Model Fine-Tuning")
plt.savefig(f"{output_path}/f1_and_accuracy.png", dpi=300)

print("Graphs saved to disk")
