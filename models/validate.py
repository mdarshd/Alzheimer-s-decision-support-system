"""
models/validate.py

Standalone evaluation script for a trained MRI stage classifier. Loads a
saved ResNet18 checkpoint and reports accuracy, a classification report,
and a confusion matrix against a labeled test image folder.

Run with: python -m models.validate

Note: NUM_CLASSES below is left at its original value of 2. The model
checkpoints in this project were trained with 4 output classes (see
detection/mri_detection.py) — update NUM_CLASSES to 4 if you're
evaluating one of those checkpoints, or point MODEL_PATH at a genuinely
binary classifier.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "mri_resnet18_stage.pth")
TEST_DIR = os.path.join(BASE_DIR, "data", "mri", "test")  # change to your test dataset path
BATCH_SIZE = 16
NUM_CLASSES = 2  # change if evaluating a multiclass checkpoint (see note above)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_test_data():
    """Load the labeled test image folder into a DataLoader."""
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=TEST_TRANSFORMS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_dataset, test_loader


def load_model():
    """Load a ResNet18 checkpoint with a NUM_CLASSES-sized output head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def evaluate(model, test_dataset, test_loader):
    """Run the model over the test set and print accuracy, report, and confusion matrix."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {accuracy:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


def main():
    """Load the model and test data, then run and print the evaluation."""
    test_dataset, test_loader = load_test_data()
    model = load_model()
    evaluate(model, test_dataset, test_loader)


if __name__ == "__main__":
    main()
