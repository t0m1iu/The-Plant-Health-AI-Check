import os
import random
import shutil
from pathlib import Path

# ==============================
# SETTINGS
# ==============================

SOURCE = Path(
    "dataset/raw/PlantVillage-Dataset/raw/color"
)

DESTINATION = Path("dataset")

TOTAL_HEALTHY = 2000
TOTAL_DISEASED = 2000

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)

# ==============================
# FIND IMAGES
# ==============================

healthy_images = []
diseased_images = []

print("Searching PlantVillage color dataset...")

for class_dir in SOURCE.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    images = []

    for file in class_dir.iterdir():

        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            images.append(file)

    if "___healthy" in class_name:
        healthy_images.extend(images)

    else:
        diseased_images.extend(images)

print()
print("Found:")
print(f"Healthy images:  {len(healthy_images)}")
print(f"Diseased images: {len(diseased_images)}")

# ==============================
# CHECK DATASET SIZE
# ==============================

if len(healthy_images) < TOTAL_HEALTHY:
    raise RuntimeError(
        f"Not enough healthy images. "
        f"Found {len(healthy_images)}"
    )

if len(diseased_images) < TOTAL_DISEASED:
    raise RuntimeError(
        f"Not enough diseased images. "
        f"Found {len(diseased_images)}"
    )

# ==============================
# RANDOM SELECTION
# ==============================

random.shuffle(healthy_images)
random.shuffle(diseased_images)

healthy_images = healthy_images[:TOTAL_HEALTHY]
diseased_images = diseased_images[:TOTAL_DISEASED]

# ==============================
# SPLIT FUNCTION
# ==============================

def split_images(images):

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


healthy_train, healthy_val, healthy_test = split_images(
    healthy_images
)

diseased_train, diseased_val, diseased_test = split_images(
    diseased_images
)

# ==============================
# COPY FUNCTION
# ==============================

def copy_images(images, destination, prefix):

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(images):

        extension = source.suffix.lower()

        destination_file = (
            destination /
            f"{prefix}_{index:05d}{extension}"
        )

        shutil.copy2(
            source,
            destination_file
        )

# ==============================
# CREATE DATASET
# ==============================

print()
print("Creating dataset...")
print()

copy_images(
    healthy_train,
    DESTINATION / "train" / "healthy",
    "healthy"
)

copy_images(
    healthy_val,
    DESTINATION / "val" / "healthy",
    "healthy"
)

copy_images(
    healthy_test,
    DESTINATION / "test" / "healthy",
    "healthy"
)

copy_images(
    diseased_train,
    DESTINATION / "train" / "diseased",
    "diseased"
)

copy_images(
    diseased_val,
    DESTINATION / "val" / "diseased",
    "diseased"
)

copy_images(
    diseased_test,
    DESTINATION / "test" / "diseased",
    "diseased"
)

# ==============================
# RESULTS
# ==============================

print("================================")
print("DATASET CREATED")
print("================================")

print()
print("TRAIN")
print(
    f"Healthy:  {len(healthy_train)}"
)
print(
    f"Diseased: {len(diseased_train)}"
)

print()
print("VALIDATION")
print(
    f"Healthy:  {len(healthy_val)}"
)
print(
    f"Diseased: {len(diseased_val)}"
)

print()
print("TEST")
print(
    f"Healthy:  {len(healthy_test)}"
)
print(
    f"Diseased: {len(diseased_test)}"
)

print()
print("Dataset location:")
print(DESTINATION.resolve())

print()
print("Done!")