# %% [markdown]
# # Imports
import sys

if len(sys.argv) == 1:
    output_path = "."
elif len(sys.argv) == 2:
    output_path = sys.argv[1]
else:
    print("Usage: python test_graph_saving.py [output_path]")
    sys.exit(1)

# %%
# Dependencies
#!pip install medmnist

# %%
import numpy as np

# %matplotlib notebook
# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import warnings

# warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
import medmnist
from medmnist import INFO, Evaluator

# %% [markdown]
# # Data Preprocessing

# %%
data_flag = "organamnist"
download = True

NUM_EPOCHS = 8
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])

DataClass = getattr(medmnist, info["python_class"])

# %%
train_dataset = DataClass(split="train", download=True, transform=transforms.ToTensor())
train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=False)


# %%
def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(dim=1).sum()
        std += images.std(dim=1).sum()
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std


# Calculate mean and std
mean, std = compute_mean_std(train_loader)
print(f"Mean: {mean}, Std: {std}")

# %%
data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.468], std=[0.2472])]
)

train_dataset = DataClass(split="train", transform=data_transform, download=download)
test_dataset = DataClass(split="test", transform=data_transform, download=download)

pil_dataset = DataClass(split="train", download=download)

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

# %%
print(train_dataset)
print("===================")
print(test_dataset)

# %% [markdown]
# # Baseline MLP


# %%
def dataloader_to_numpy(data_loader):
    data_list = []
    label_list = []
    for data, labels in data_loader:
        data_list.append(data)  # Assuming tensors
        label_list.append(labels)
    return np.concatenate(data_list), np.concatenate(label_list)


# Convert train and test loaders
X_train, y_train = dataloader_to_numpy(train_dataset)
X_test, y_test = dataloader_to_numpy(test_dataset)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# %%
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 256), max_iter=300, activation="relu", solver="adam"
)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# %% [markdown]
# # CNN model


# %%
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        conv1_output_size = self.calculate_output_dim(
            28, kernel_size=5, stride=1, padding=1
        )
        conv2_output_size = self.calculate_output_dim(
            conv1_output_size, kernel_size=3, stride=1, padding=1
        )

        # Fully connected layers
        fc_input_dim = conv2_output_size * conv2_output_size * 64
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def calculate_output_dim(self, input_size, kernel_size, stride, padding):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# %% [markdown]
# # Experiments
#

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 11
model = CNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # Forward pass
        labels = labels.view(-1).long()
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()  # Set the model to evaluation mode
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)
            labels = labels.view(-1).long()
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = running_loss_test / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
    )

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Training and Test Accuracy")

plt.tight_layout()
plt.savefig(f"{output_path}/plot_28x28.png", dpi=300, bbox_inches="tight")

# %%


# %% [markdown]
# # (Creativity) Effects of model hyperparameters on accuracy

# %%


# %% [markdown]
# # 128 pixel version


# %%
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        conv1_output_size = self.calculate_output_dim(
            128, kernel_size=5, stride=1, padding=1
        )
        conv2_output_size = self.calculate_output_dim(
            conv1_output_size, kernel_size=3, stride=1, padding=1
        )

        # Fully connected layers
        fc_input_dim = conv2_output_size * conv2_output_size * 64
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def calculate_output_dim(self, input_size, kernel_size, stride, padding):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# %%
data_flag = "organamnist"
download = True

NUM_EPOCHS = 8
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])

DataClass = getattr(medmnist, info["python_class"])

# %%
data_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.468], std=[0.2349])]
)

train_dataset = DataClass(
    split="train", size=128, transform=data_transform, download=download
)
test_dataset = DataClass(
    split="test", transform=data_transform, size=128, download=download
)

pil_dataset = DataClass(split="train", download=download)

train_loader = data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
train_loader_at_eval = data.DataLoader(
    dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)
test_loader = data.DataLoader(
    dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
)

# %%
mean, std = compute_mean_std(train_loader)
print(f"Mean: {mean}, Std: {std}")

# %%
print(train_dataset)
print("===================")
print(test_dataset)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 11
model = CNN(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
NUM_EPOCHS = 8
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to device
        optimizer.zero_grad()  # Clear gradients
        outputs = model(images)  # Forward pass
        labels = labels.view(-1).long()
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    model.eval()  # Set the model to evaluation mode
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)
            labels = labels.view(-1).long()
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_loss = running_loss_test / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
        f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%"
    )

# %%
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Training and Test Accuracy")

plt.tight_layout()
plt.savefig(f"{output_path}/plot_128x128.png", dpi=300, bbox_inches="tight")
