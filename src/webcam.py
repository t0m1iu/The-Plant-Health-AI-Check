import cv2
import torch

from torchvision import transforms, models
from torch import nn


# ============================================================
# PLANT HEALTH AI - LIVE WEBCAM
# Jetson Orin Nano
# ResNet-18
# ============================================================


MODEL_PATH = (
    "/workspace/plant-health-ai/"
    "models/resnet18_plant_combined.pth"
)


IMAGE_SIZE = 224

CAMERA_DEVICE = "/dev/video0"

# Minimum confidence required before showing a warning
DISEASE_THRESHOLD = 0.70


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print()
print("=" * 60)
print("PLANT HEALTH AI - LIVE WEBCAM")
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

print()
print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


classes = checkpoint["classes"]

print(
    "Classes:",
    classes
)


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
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.ToPILImage(),

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
# CAMERA
# ============================================================

print()
print("Opening camera...")


camera = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)


if not camera.isOpened():

    print()
    print("ERROR: Could not open camera.")

    print()
    print("Try changing:")
    print(
        'CAMERA_DEVICE = "/dev/video0"'
    )

    print(
        "to /dev/video1"
    )

    raise SystemExit


# Camera resolution

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


camera.set(
    cv2.CAP_PROP_FPS,
    30
)


print()
print("Camera opened successfully.")

print()
print("Controls:")
print("  Q = Quit")
print("  ESC = Quit")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print(
            "ERROR: Could not read camera frame."
        )

        break


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = transform(
        rgb
    )


    image = image.unsqueeze(
        0
    )


    image = image.to(
        DEVICE
    )


    # --------------------------------------------------------
    # AI prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image
        )


        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    predicted_index = (
        probabilities.argmax().item()
    )


    predicted_class = classes[
        predicted_index
    ]


    confidence = (
        probabilities[
            predicted_index
        ].item()
    )


    # --------------------------------------------------------
    # Find probabilities
    # --------------------------------------------------------

    diseased_probability = 0.0

    healthy_probability = 0.0


    for i, class_name in enumerate(classes):

        probability = (
            probabilities[i].item()
        )


        if class_name == "diseased":

            diseased_probability = probability


        elif class_name == "healthy":

            healthy_probability = probability


    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    if (
        predicted_class == "diseased"
        and
        diseased_probability >= DISEASE_THRESHOLD
    ):

        status = "WARNING: POSSIBLE DISEASE"

        status_color = (
            0,
            0,
            255
        )

    else:

        status = "HEALTHY - NO WARNING"

        status_color = (
            0,
            200,
            0
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display = frame.copy()


    # Top information panel

    cv2.rectangle(
        display,
        (0, 0),
        (1280, 150),
        (20, 20, 20),
        -1
    )


    # Title

    cv2.putText(
        display,
        "PLANT HEALTH AI",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )


    # Status

    cv2.putText(
        display,
        status,
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        status_color,
        2
    )


    # Confidence

    cv2.putText(
        display,
        f"Confidence: {confidence * 100:.2f}%",
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # Probability information on right

    cv2.putText(
        display,
        f"Diseased: {diseased_probability * 100:.1f}%",
        (850, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        f"Healthy: {healthy_probability * 100:.1f}%",
        (850, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display,
        "Press Q to quit",
        (850, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1
    )


    # --------------------------------------------------------
    # Warning border
    # --------------------------------------------------------

    if (
        predicted_class == "diseased"
        and
        diseased_probability >= DISEASE_THRESHOLD
    ):

        cv2.rectangle(
            display,
            (5, 5),
            (
                display.shape[1] - 5,
                display.shape[0] - 5
            ),
            (0, 0, 255),
            8
        )


    # --------------------------------------------------------
    # Show image
    # --------------------------------------------------------

    cv2.imshow(
        "Plant Health AI",
        display
    )


    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    if key == 27:

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()
print("Webcam stopped.")
