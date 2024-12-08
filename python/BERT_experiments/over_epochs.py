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
from sklearn.metrics import classification_report

if len(sys.argv) == 1:
    output_path = "."
    train_method = "full"
    num_epochs = 10
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
    train_method = "full"
elif len(sys.argv) == 3:
    output_path = sys.argv[1]
    if str.lower(sys.argv[2]) == "head":
        train_method = "head"
    elif str.lower(sys.argv[2]) == "full":
        train_method = "full"
    elif str.lower(sys.argv[2]) == "head+1":
        train_method = "head+1"
    else:
        print("Invalid train method. Options: head, full, head+1")
        sys.exit(1)
    num_epochs = 10
elif len(sys.argv) == 4:
    output_path = sys.argv[1]
    if str.lower(sys.argv[2]) == "head":
        train_method = "head"
    elif str.lower(sys.argv[2]) == "full":
        train_method = "full"
    elif str.lower(sys.argv[2]) == "head+1":
        train_method = "head+1"
    else:
        print("Invalid train method. Options: head, full, head+1")
        sys.exit(1)
    num_epochs = int(sys.argv[3])
else:
    print(
        "Usage: python over_epochs.py [output_path] [train_method (head, full, head+1)] [num_epochs]"
    )
    sys.exit(1)

print("Starting BERT epoch experiments script")
print("Train method:", train_method)

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

label_strings = []
for key in range(28):
    label_strings.append(label_map[str(key)])

# Path to the local directory containing the saved model
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"

model_path = bert_path

# Set the hyperparameters
batch_size = 32
weight_decay = 1
warmup_steps = 500
# Set the learning rate and number of epochs depending on head or full fine-tune
if train_method == "head":
    learning_rate = 0.01
elif train_method == "full":
    learning_rate = 2e-5
elif train_method == "head+1":
    learning_rate = 1e-3

print(
    "Learning rate:",
    learning_rate,
    "\nNum epochs:",
    num_epochs,
    "\nBatch size:",
    batch_size,
    "\nWeight decay:",
    weight_decay,
    "\nWarmup steps:",
    warmup_steps,
)

# Initialize the results dictionary
results = {"f1": [], "accuracy": [], "duration": [], "reports": []}

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

if train_method == "head":
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # Only the classification head will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True
elif train_method == "head+1":
    # Freeze all BERT layers
    for param in model.bert.parameters():
        param.requires_grad = False

    # Only the classification head will be trained
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze the last transformer layer
    for param in model.bert.encoder.layer[11].parameters():
        param.requires_grad = True

    # Unfreeze the pooler layer
    for param in model.bert.pooler.parameters():
        param.requires_grad = True

for name, param in model.named_parameters():
    print(
        "Name:",
        name,
        " - Size:",
        param.size(),
        " - Requires grad:",
        param.requires_grad,
    )

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
    f1 = f1_score(labels, predictions, average="macro", zero_division=0.0)
    report = classification_report(
        labels,
        predictions,
        target_names=label_strings,
    )
    results["f1"].append(f1)
    results["accuracy"].append(accuracy)
    results["duration"].append(time.time() - start_time)
    results["reports"].append(report)
    return {
        "accuracy": accuracy,
        "f1": f1,
        "classification_report": report,
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
    warmup_steps=warmup_steps,  # Number of warmup steps for learning rate scheduler
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

# Step 6: Train & save the model
trainer.train()

# Group unfrozen layers
layers_to_save = {}
if train_method == "head":
    print("Saving classifier layer")
    layers_to_save["classifier"] = model.classifier.state_dict()
elif train_method == "head+1":
    print("Saving classifier, encoder, and pooler layers")
    layers_to_save["classifier"] = model.classifier.state_dict()
    layers_to_save["encoder"] = model.bert.encoder.layer[11].state_dict()
    layers_to_save["pooler"] = model.bert.pooler.state_dict()
elif train_method == "full":
    print("Saving all layers")
    layers_to_save["model"] = model.state_dict()

print("Gathered layers to save.")

torch.save(layers_to_save, f"{output_path}/selected_layers_state_dict.pth")
print("Layer weights saved.")

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


def get_instances_by_prediction_correctness(
    dataset, preds, class_preds, correct=True, num_instances=10
):
    assert dataset.num_rows == len(preds.label_ids) == len(class_preds)
    p = np.random.permutation(dataset.num_rows)

    # Collect instances
    instances = []
    i = 0
    while len(instances) < num_instances and i < len(p):
        idx = p[i]

        if (correct and preds.label_ids[idx] == class_preds[idx]) or (
            not correct and preds.label_ids[idx] != class_preds[idx]
        ):
            # Save instance number, tokens, true label, and predicted label (labels as strings)
            instances.append(
                [
                    idx,
                    dataset["text"][idx],
                    label_map[str(preds.label_ids[idx][0])],
                    label_map[str(class_preds[idx])],
                ]
            )
            print(instances[-1])

        i += 1

    return instances


correct_instances = get_instances_by_prediction_correctness(
    tokenized_datasets["test"], predictions, class_predictions, correct=True
)

incorrect_instances = get_instances_by_prediction_correctness(
    tokenized_datasets["test"], predictions, class_predictions, correct=False
)

print("Correct Instances: ", correct_instances)
print("Incorrect Instances: ", incorrect_instances)


f1 = f1_score(
    predictions.label_ids, class_predictions, average="macro", zero_division=0.0
)
accuracy = accuracy_score(predictions.label_ids, class_predictions)
print(
    "Metrics:\nF1:",
    f1,
    "\nAccuracy:",
    accuracy,
)

duration = time.time() - start_time
report = classification_report(
    predictions.label_ids,
    class_predictions,
    target_names=label_strings,
)


results["final"] = {}
results["final"]["f1"] = f1
results["final"]["accuracy"] = accuracy
results["final"]["duration"] = duration
results["final"]["report"] = report

print("Results:", results)
print("Final Report:\n", report)

epoch_list = range(num_epochs + 1)
f1s = results["f1"]
accuracies = results["accuracy"]
durations = results["duration"]

print("F1 scores:", f1s)
print("Accuracies:", accuracies)
print("Durations:", durations)

# Access loss history

log_history = trainer.state.log_history

# Extract training losses
training_losses = [entry["loss"] for entry in log_history if "loss" in entry]

# Extract validation losses
validation_losses = [
    entry["eval_loss"] for entry in log_history if "eval_loss" in entry
]

# Print or use the results
print("log_history:", log_history)
print("Training Losses (", len(training_losses), "):", training_losses)
print("Validation Losses (", len(validation_losses), "):", validation_losses)


# Graph the results
train_method_title = train_method.title()

plt.figure()
plt.plot(epoch_list, f1s, label="F1 Score")
plt.title(f"F1 Score Over Epochs for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.savefig(f"{output_path}/f1.png", dpi=300)

plt.figure()
plt.plot(epoch_list, accuracies, label="Accuracy")
plt.title(f"Accuracy Over Epochs for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig(f"{output_path}/accuracy.png", dpi=300)

plt.figure()
plt.plot(epoch_list, durations, label="Duration")
plt.title(f"Duration Over Epochs for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Duration (s)")
plt.savefig(f"{output_path}/duration.png", dpi=300)

plt.figure()
plt.plot(epoch_list, f1s, label="F1 Score")
plt.plot(epoch_list, accuracies, label="Accuracy")
plt.title(f"F1 Score and Accuracy Over Epochs for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("F1 Score/Accuracy")
plt.savefig(f"{output_path}/f1_and_accuracy.png", dpi=300)

plt.figure()
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.title(f"Loss Over Epochs for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f"{output_path}/losses.png", dpi=300)

print("Graphs saved to disk")
