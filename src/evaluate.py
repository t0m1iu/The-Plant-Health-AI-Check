import os
import torch

from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "/workspace/plant-health-ai/"
    "models/resnet18_plant_combined.pth"
)

DATASET_DIR = (
    "/workspace/plant-health-ai/"
    "dataset/combined"
)

IMAGE_SIZE = 224

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 60)
print("PLANT HEALTH AI - MODEL EVALUATION")
print("=" * 60)

print()

print("Device:", DEVICE)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

print()
print("Classes:", classes)

model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(
    DEVICE
)

model.eval()


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# EVALUATION
# ============================================================

def evaluate_directory(directory):

    dataset = datasets.ImageFolder(
        directory,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    correct = 0

    total = 0

    confusion = torch.zeros(
        2,
        2,
        dtype=torch.int64
    )


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

            predictions = outputs.argmax(
                dim=1
            )


            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


            for actual, predicted in zip(
                labels.cpu(),
                predictions.cpu()
            ):

                confusion[
                    actual,
                    predicted
                ] += 1


    accuracy = (
        correct / total
        if total > 0
        else 0
    )


    return (
        accuracy,
        confusion
    )


# ============================================================
# PLANTVILLAGE TEST
# ============================================================

print()
print("=" * 60)
print("PLANTVILLAGE TEST")
print("=" * 60)

pv_accuracy, pv_confusion = evaluate_directory(
    os.path.join(
        DATASET_DIR,
        "test"
    )
)

print()
print(
    f"Accuracy: "
    f"{pv_accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("Confusion Matrix")
print()
print("Rows = Actual")
print("Columns = Predicted")
print()

print(
    "                 Diseased    Healthy"
)

print(
    f"Diseased       "
    f"{pv_confusion[0,0]:8d} "
    f"{pv_confusion[0,1]:10d}"
)

print(
    f"Healthy        "
    f"{pv_confusion[1,0]:8d} "
    f"{pv_confusion[1,1]:10d}"
)


# ============================================================
# CLASS METRICS
# ============================================================

diseased_correct = (
    pv_confusion[0, 0].item()
)

diseased_total = (
    pv_confusion[0].sum().item()
)

healthy_correct = (
    pv_confusion[1, 1].item()
)

healthy_total = (
    pv_confusion[1].sum().item()
)


diseased_recall = (
    diseased_correct / diseased_total
    if diseased_total
    else 0
)

healthy_recall = (
    healthy_correct / healthy_total
    if healthy_total
    else 0
)


print()

print(
    f"Diseased Recall: "
    f"{diseased_recall * 100:.2f}%"
)

print(
    f"Healthy Recall: "
    f"{healthy_recall * 100:.2f}%"
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
