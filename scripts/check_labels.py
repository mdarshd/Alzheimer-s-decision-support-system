"""
scripts/check_labels.py

Small utility that prints the class-to-index mapping torchvision's
ImageFolder assigns to the MRI train/test directories. Useful for
sanity-checking that ORIGINAL_LABELS in app.py / detection/mri_detection.py
still matches the folder structure of data/mri.

Run with: python -m scripts.check_labels
"""

from torchvision import datasets


def main():
    """Print the ImageFolder class-to-index mapping for the train and test sets."""
    train_ds = datasets.ImageFolder("data/mri/train")
    test_ds = datasets.ImageFolder("data/mri/test")

    print("Train mapping:", train_ds.class_to_idx)
    print("Test mapping :", test_ds.class_to_idx)


if __name__ == "__main__":
    main()
