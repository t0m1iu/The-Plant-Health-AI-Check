import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "/workspace/plant-health-ai/dataset"
MODEL_DIR = "/workspace/plant-health-ai/models"

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("PLANT HEALTH AI - RESNET-18 TRAINING")
print("=" * 60)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print()


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "train"),
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "val"),
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_DIR, "test"),
    transform=val_test_transforms
)


print()
print("Classes:", train_dataset.classes)

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)

print()


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# ============================================================
# RESNET-18
# ============================================================

print("Loading pretrained ResNet-18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# Replace final classification layer

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    2
)


model = model.to(DEVICE)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_accuracy = 0.0

best_model_weights = copy.deepcopy(
    model.state_dict()
)


for epoch in range(NUM_EPOCHS):

    print()
    print("=" * 60)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 60)

    start_time = time.time()


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad()


        outputs = model(images)


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item() * images.size(0)
        )


        _, predictions = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100.0 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += (
                loss.item() * images.size(0)
            )


            _, predictions = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    validation_loss = (
        val_loss / val_total
    )

    validation_accuracy = (
        100.0 * val_correct / val_total
    )


    elapsed = time.time() - start_time


    print()
    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss:         {validation_loss:.4f}"
    )

    print(
        f"Val Accuracy:     {validation_accuracy:.2f}%"
    )

    print(
        f"Time:             {elapsed:.1f} seconds"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if validation_accuracy > best_accuracy:

        best_accuracy = validation_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )


        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )


        model_path = os.path.join(
            MODEL_DIR,
            "resnet18_plant.pth"
        )


        torch.save(
            {
                "model_state_dict":
                    best_model_weights,

                "classes":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "validation_accuracy":
                    validation_accuracy
            },
            model_path
        )


        print()
        print(
            "NEW BEST MODEL SAVED!"
        )

        print(
            model_path
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 60)
print("FINAL TEST")
print("=" * 60)


model.eval()

test_correct = 0
test_total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        outputs = model(images)


        _, predictions = torch.max(
            outputs,
            1
        )


        test_total += labels.size(0)

        test_correct += (
            predictions == labels
        ).sum().item()


test_accuracy = (
    100.0 * test_correct / test_total
)


print()
print(
    f"Test Accuracy: {test_accuracy:.2f}%"
)

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print()
print(
    "Model saved to:"
)

print(
    os.path.join(
        MODEL_DIR,
        "resnet18_plant.pth"
    )
)

print()

