import os
import time
import copy
import torch

from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "/workspace/plant-health-ai/dataset/combined"

MODEL_DIR = "/workspace/plant-health-ai/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_plant_combined.pth"
)

BATCH_SIZE = 32

NUM_EPOCHS = 12

IMAGE_SIZE = 224

LEARNING_RATE = 0.0001

NUM_WORKERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("PLANT HEALTH AI")
print("PLANTVILLAGE + PLANTDOC")
print("RESNET-18 FINE-TUNING")
print("=" * 60)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.75, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        20
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


eval_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASETS
# ============================================================

print()
print("Loading datasets...")


train_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "train"
    ),
    transform=train_transform
)


val_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "val"
    ),
    transform=eval_transform
)


test_dataset = datasets.ImageFolder(
    os.path.join(
        DATASET_DIR,
        "test"
    ),
    transform=eval_transform
)


print()

print(
    "Classes:",
    train_dataset.classes
)

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


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = []

for class_name in train_dataset.classes:

    class_directory = os.path.join(
        DATASET_DIR,
        "train",
        class_name
    )

    count = len([
        f for f in os.listdir(class_directory)
        if os.path.isfile(
            os.path.join(
                class_directory,
                f
            )
        )
    ])

    class_counts.append(count)


print()
print("Class counts:")

for name, count in zip(
    train_dataset.classes,
    class_counts
):

    print(
        f"{name}: {count}"
    )


# Higher weight for smaller class

total = sum(class_counts)

class_weights = [
    total / (2 * count)
    for count in class_counts
]


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print()
print("Class weights:")

print(
    class_weights
)


# ============================================================
# MODEL
# ============================================================

print()
print("Loading pretrained ResNet-18...")


model = models.resnet18(
    weights=ResNet18_Weights.DEFAULT
)


# Replace classifier

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0

    correct = 0

    total_samples = 0


    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = outputs.argmax(
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total_samples += (
            images.size(0)
        )


    return (
        running_loss / total_samples,
        correct / total_samples
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate(loader):

    model.eval()

    running_loss = 0.0

    correct = 0

    total_samples = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            running_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()


            total_samples += (
                images.size(0)
            )


    return (
        running_loss / total_samples,
        correct / total_samples
    )


# ============================================================
# TRAIN
# ============================================================

best_val_accuracy = 0.0

best_state = None


print()
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(
    NUM_EPOCHS
):

    start_time = time.time()


    train_loss, train_accuracy = (
        train_one_epoch()
    )


    val_loss, val_accuracy = (
        evaluate(
            val_loader
        )
    )


    scheduler.step(
        val_accuracy
    )


    elapsed = (
        time.time()
        - start_time
    )


    print()

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss:      {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:  "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:        {val_loss:.4f}"
    )

    print(
        f"Val Accuracy:    "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Time:            "
        f"{elapsed:.1f}s"
    )


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = (
            val_accuracy
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )


        torch.save(
            {
                "model_state_dict":
                    best_state,

                "classes":
                    train_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "val_accuracy":
                    best_val_accuracy
            },
            MODEL_PATH
        )


        print()
        print(
            "NEW BEST MODEL SAVED!"
        )

        print(
            MODEL_PATH
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy = (
    evaluate(
        test_loader
    )
)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print()

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Combined Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print()

print(
    "Model saved to:"
)

print(
    MODEL_PATH
)

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
