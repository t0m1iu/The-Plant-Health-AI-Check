import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


MODEL_PATH = "/workspace/plant-health-ai/models/resnet18_plant.pth"

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# CHECK ARGUMENT
# ------------------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("python3 src/predict.py IMAGE_PATH")
    sys.exit(1)


image_path = sys.argv[1]


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

print("Classes:", classes)


model = models.resnet18(
    weights=None
)

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ------------------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------------------
# LOAD IMAGE
# ------------------------------------------------------------

try:

    image = Image.open(
        image_path
    ).convert("RGB")

except Exception as e:

    print("ERROR: Could not open image.")
    print(e)

    sys.exit(1)


image_tensor = transform(
    image
).unsqueeze(0)


image_tensor = image_tensor.to(
    DEVICE
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        1
    )


predicted_class = classes[
    prediction.item()
]

confidence_percent = (
    confidence.item() * 100
)


# ------------------------------------------------------------
# RESULT
# ------------------------------------------------------------

print()
print("=" * 50)
print("PLANT HEALTH CHECK")
print("=" * 50)

print(
    f"Prediction: {predicted_class.upper()}"
)

print(
    f"Confidence: {confidence_percent:.2f}%"
)

print()

for i, class_name in enumerate(classes):

    print(
        f"{class_name.capitalize():10s}: "
        f"{probabilities[0][i].item() * 100:.2f}%"
    )

print("=" * 50)

if predicted_class == "diseased":

    print()
    print("WARNING: Possible unhealthy leaf detected.")

else:

    print()
    print("Leaf appears healthy.")

print()
