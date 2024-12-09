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


# Function to choose what weights to train
def get_train_method(str):
    if str.lower() == "head":
        return "head"
    elif str.lower() == "full":
        return "full"
    elif str.lower() == "head+1":
        return "head+1"
    else:
        print("Invalid train method. Options: head, full, head+1")
        sys.exit(1)


# Function to choose how to search hyperparameters
def get_search_hyperparam(str, values=None):
    if str.lower() == "default":
        str = "weight_decay"

    if values is not None:
        values = values.split(",")
        search_space = [float(i) for i in values]

    if str.lower() == "batch_size":
        if values is None:
            search_space = [8, 16, 32, 64, 128, 256]
        else:
            search_space = [int(i) for i in values]
        return "batch_size", search_space
    elif str.lower() == "weight_decay":
        if values is None:
            search_space = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        return "weight_decay", search_space
    elif str.lower() == "learning_rate":
        if values is None:
            search_space = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        return "learning_rate", search_space
    else:
        print(
            "Invalid search hyperparam. Options: batch_size, weight_decay, learning_rate"
        )
        sys.exit(1)


# Check for command line arguments
if len(sys.argv) == 1:
    output_path = "."
    train_method = "full"
    search_hyperparam, search_space = get_search_hyperparam("default")
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
    train_method = "full"
    search_hyperparam, search_space = get_search_hyperparam("default")
elif len(sys.argv) == 3:
    output_path = sys.argv[1]
    train_method = get_train_method(sys.argv[2])
    search_hyperparam, search_space = get_search_hyperparam("default")
elif len(sys.argv) == 4:
    output_path = sys.argv[1]
    train_method = get_train_method(sys.argv[2])
    search_hyperparam, search_space = get_search_hyperparam(sys.argv[3])
elif len(sys.argv) == 5:
    output_path = sys.argv[1]
    train_method = get_train_method(sys.argv[2])
    search_hyperparam, search_space = get_search_hyperparam(
        sys.argv[3], values=sys.argv[4]
    )
else:
    print(
        "Usage: python over_epochs.py [output_path] [train_method (head, full, head+1)] [search_hyperparam (batch_size, weight_decay, learning_rate)] [values (optional)]"
    )
    sys.exit(1)

print("Starting BERT epoch experiments script")
print("Train method:", train_method)

# Define the label map
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

# Path to the local directory containing the saved model (for SLURM)
bert_path = "/opt/models/bert-base-uncased"
distil_path = "/opt/models/distilgpt2"
model_path = bert_path

# Set the hyperparameters
num_epochs = 8
batch_size = 16
weight_decay = 0.5
warmup_steps = 500
# Set the learning rate and number of epochs depending on head or full fine-tune
if train_method == "head":
    learning_rate = 0.01
elif train_method == "full":
    learning_rate = 2e-5
elif train_method == "head+1":
    learning_rate = 1e-3

print("======= Search hyperparam:", search_hyperparam, " =======")


overall_results = {}

# Loop over the search space
for hyperparam in search_space:
    print("==== ", search_hyperparam, ": ", hyperparam, " ====")
    if search_hyperparam == "batch_size":
        batch_size = hyperparam
    elif search_hyperparam == "weight_decay":
        weight_decay = hyperparam
    elif search_hyperparam == "learning_rate":
        learning_rate = hyperparam

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

    # Send to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model.to(device)

    # Freeze layers depending on which are selected for training
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

    # Print the trainable parameters
    """for name, param in model.named_parameters():    
        print(
            "Name:",
            name,
            " - Size:",
            param.size(),
            " - Requires grad:",
            param.requires_grad,
        ) """

    # Load the GoEmotions dataset
    print("Fine-tuning the model on the GoEmotions dataset...")
    dataset = load_dataset("google-research-datasets/go_emotions")

    # Filter for examples with a single label
    def filter_single_label(example):
        return len(example["labels"]) == 1

    filtered_dataset = dataset.filter(filter_single_label)
    print("Filtered dataset:\n", filtered_dataset)

    # Preprocess the dataset (tokenize)
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    tokenized_datasets = filtered_dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Define the evaluation metrics
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
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
        }

    # Set up training arguments
    # Note: set to give minimal logs and no saving, to preserve disk space on MIMI
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

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Test the model
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
    report = classification_report(
        predictions.label_ids,
        class_predictions,
        target_names=label_strings,
    )

    # Save results
    results["final"] = {}
    results["final"]["f1"] = f1
    results["final"]["accuracy"] = accuracy
    results["final"]["duration"] = duration
    results["final"]["report"] = report

    # print("Results:", results)
    print("Final Report:\n", report)

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
    # print("log_history:", log_history)
    print("Training Losses (", len(training_losses), "):", training_losses)
    print("Validation Losses (", len(validation_losses), "):", validation_losses)

    results["final"]["training_losses"] = training_losses
    results["final"]["validation_losses"] = validation_losses

    overall_results[hyperparam] = results

# Graph the results
train_method_title = train_method.title()
if search_hyperparam == "learning_rate" or search_hyperparam == "weight_decay":
    x_scale = "log"
else:
    x_scale = "linear"

# F1 score
plt.figure()
f1s = []
for hyperparam in search_space:
    f1s.append(overall_results[hyperparam]["final"]["f1"])
plt.plot(search_space, f1s, label=f"F1 Score")
plt.title(f"F1 Score Over {search_hyperparam} for {train_method_title} Fine-Tuning")
plt.legend(title=f"{search_hyperparam} values:")
plt.xlabel(f"{search_hyperparam}")
plt.ylabel("F1 Score")
plt.xscale(x_scale)
plt.savefig(f"{output_path}/f1.png", dpi=300)

# Accuracy
plt.figure()
accuracies = []
for hyperparam in search_space:
    accuracies.append(overall_results[hyperparam]["final"]["accuracy"])
plt.plot(search_space, accuracies, label=f"Acuracy")
plt.title(f"Accuracy Over {search_hyperparam} for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel(f"{search_hyperparam}")
plt.ylabel("Accuracy")
plt.xscale(x_scale)
plt.savefig(f"{output_path}/accuracy.png", dpi=300)

# Duration
plt.figure()
durations = []
for hyperparam in search_space:
    durations.append(overall_results[hyperparam]["final"]["duration"])
plt.plot(search_space, durations, label=f"Duration")
plt.title(f"Duration Over {search_hyperparam} for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel(f"{search_hyperparam}")
plt.ylabel("Duration (s)")
plt.xscale(x_scale)
plt.savefig(f"{output_path}/duration.png", dpi=300)

# F1 and Accuracy
plt.figure()
plt.plot(search_space, f1s, label="F1 Score")
plt.plot(search_space, accuracies, label="Accuracy")
plt.title(
    f"F1 Score and Accuracy Over {search_hyperparam} for {train_method_title} Fine-Tuning"
)
plt.legend()
plt.xlabel(f"{search_hyperparam}")
plt.ylabel("F1 Score/Accuracy")
plt.xscale(x_scale)
plt.savefig(f"{output_path}/f1_and_accuracy.png", dpi=300)

# Losses
plt.figure()
training_losses = []
validation_losses = []
for hyperparam in search_space:
    training_losses.append(
        overall_results[hyperparam]["final"]["training_losses"][
            len(overall_results[hyperparam]["final"]["training_losses"]) - 1
        ]
    )
    validation_losses.append(
        overall_results[hyperparam]["final"]["validation_losses"][
            len(overall_results[hyperparam]["final"]["validation_losses"]) - 1
        ]
    )
plt.plot(search_space, training_losses, label=f"Training Loss")
plt.plot(search_space, validation_losses, label=f"Validation Loss")
plt.title(f"Loss Over {search_hyperparam} for {train_method_title} Fine-Tuning")
plt.legend()
plt.xlabel(f"{search_hyperparam}")
plt.ylabel("Loss")
plt.xscale(x_scale)
plt.savefig(f"{output_path}/losses.png", dpi=300)

print("Graphs saved to disk")
