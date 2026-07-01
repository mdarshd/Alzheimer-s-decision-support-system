"""
detection/mri_detection.py

Standalone training script for the MRI-based impairment stage classifier.
Fine-tunes a ResNet18 on the labeled MRI image dataset (data/mri/train,
data/mri/test), evaluates it on the held-out test split, and saves the
resulting weights to models/mri_resnet18_stage.pth for use by app.py.

Run with: python -m detection.mri_detection
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "mri", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "mri", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "mri_resnet18_stage.pth")

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4

# Label ordering as produced by torchvision's ImageFolder over the raw
# dataset. Remapped to the app's clinical stage names below.
ORIGINAL_LABELS = {
    0: "Mild_Impairment",
    1: "Moderate_Impairment",
    2: "No_Impairment",
    3: "Very_Mild_Impairment",
}

UPDATED_STAGE_MAP = {
    "Mild_Impairment": "Moderate Impairment",
    "Moderate_Impairment": "Severe Impairment",
    "No_Impairment": "No Impairment",
    "Very_Mild_Impairment": "Mild Impairment",
}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_dataloaders():
    """Load the train/test MRI image folders and wrap them in DataLoaders."""
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=TRANSFORM)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=TRANSFORM)

    print("Internal ImageFolder mapping:", train_ds.class_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def build_model(device):
    """Create a ResNet18 with a fresh classification head for the impairment stages."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(device)


def train(model, train_loader, device):
    """Fine-tune the model for EPOCHS epochs, printing the average loss per epoch."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss / len(train_loader):.4f}")


def evaluate(model, test_loader, device):
    """Run the model over the test set and print a classification report."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nTEST RESULTS")
    print(classification_report(all_labels, all_preds))


def detect_single_image(model, device, image_path):
    """Classify a single MRI image and return its mapped clinical stage name."""
    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        pred_index = torch.argmax(outputs, dim=1).item()

    original_label = ORIGINAL_LABELS[pred_index]
    final_stage = UPDATED_STAGE_MAP[original_label]

    print("\nMRI DETECTION RESULT")
    print("Predicted Index :", pred_index)
    print("Original Class  :", original_label)
    print("Final Stage     :", final_stage)

    return final_stage


def main():
    """Train the MRI stage classifier end to end and save the resulting weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, test_loader = build_dataloaders()
    model = build_model(device)

    train(model, train_loader, device)
    evaluate(model, test_loader, device)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("\nModel saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
