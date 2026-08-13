import sys
import torch

from PIL import Image

from torchvision import transforms, models
from torch import nn


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = (
    "/workspace/plant-health-ai/"
    "models/resnet18_plant_combined.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]

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
# IMAGE TRANSFORM
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
# PREDICTION
# ============================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")


    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    with torch.no_grad():

        output = model(
            image_tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    prediction = probabilities.argmax().item()

    predicted_class = classes[
        prediction
    ]

    confidence = (
        probabilities[prediction].item()
        * 100
    )


    print()
    print("=" * 55)
    print("PLANT HEALTH CHECK")
    print("=" * 55)

    print()

    print(
        "Prediction:",
        predicted_class.upper()
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print()

    for i, class_name in enumerate(classes):

        print(
            f"{class_name.capitalize():10s}: "
            f"{probabilities[i].item() * 100:.2f}%"
        )

    print()

    # --------------------------------------------------------
    # WARNING LOGIC
    # --------------------------------------------------------

    if predicted_class == "diseased":

        print(
            "WARNING: Possible unhealthy leaf detected."
        )

    else:

        print(
            "OK: Leaf appears healthy."
        )

    print("=" * 55)
    print()


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) != 2:

    print(
        "Usage:"
    )

    print(
        "python3 src/predict_combined.py "
        "/path/to/image.jpg"
    )

    sys.exit(1)


predict(
    sys.argv[1]
)
