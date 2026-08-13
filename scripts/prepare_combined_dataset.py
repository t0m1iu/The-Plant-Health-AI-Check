import os
import shutil
import random
from pathlib import Path


# ============================================================
# PLANT HEALTH AI
# Combine PlantVillage + PlantDoc
#
# Output:
#
# dataset/combined/
#     train/
#         healthy/
#         diseased/
#     val/
#         healthy/
#         diseased/
#     test/
#         healthy/
#         diseased/
# ============================================================


SEED = 42

random.seed(SEED)


PROJECT_DIR = Path("/home/nvidia/plant-health-ai")

PLANTDOC_DIR = (
    PROJECT_DIR /
    "dataset/raw/PlantDoc-Dataset"
)

OUTPUT_DIR = (
    PROJECT_DIR /
    "dataset/combined"
)


# ============================================================
# PLANTDOC CATEGORIES
# ============================================================

HEALTHY_CATEGORIES = [

    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "grape leaf",
    "Peach leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
]


DISEASED_CATEGORIES = [

    "Apple rust leaf",
    "Apple Scab Leaf",
    "Bell_pepper leaf spot",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "grape leaf black rot",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Squash Powdery mildew leaf",
    "Tomato Early blight leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato Septoria leaf spot",
]


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {

    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


# ============================================================
# FUNCTIONS
# ============================================================

def get_images(directory):

    return [
        p for p in directory.rglob("*")
        if p.is_file()
        and p.suffix in IMAGE_EXTENSIONS
    ]


def copy_images(
    images,
    output_directory,
    prefix
):

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, image in enumerate(images):

        destination = (
            output_directory /
            f"{prefix}_{index:05d}{image.suffix.lower()}"
        )

        shutil.copy2(
            image,
            destination
        )


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

if OUTPUT_DIR.exists():

    print("Removing previous combined dataset...")

    shutil.rmtree(OUTPUT_DIR)


for split in [
    "train",
    "val",
    "test"
]:

    for category in [
        "healthy",
        "diseased"
    ]:

        (
            OUTPUT_DIR /
            split /
            category
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# COLLECT PLANTDOC IMAGES
# ============================================================

print()
print("=" * 60)
print("PLANT HEALTH AI DATASET PREPARATION")
print("=" * 60)

print()
print("Reading PlantDoc...")


plantdoc_train = {
    "healthy": [],
    "diseased": []
}


plantdoc_test = {
    "healthy": [],
    "diseased": []
}


for category in HEALTHY_CATEGORIES:

    directory = (
        PLANTDOC_DIR /
        "train" /
        category
    )

    if directory.exists():

        images = get_images(directory)

        plantdoc_train["healthy"].extend(
            images
        )


for category in DISEASED_CATEGORIES:

    directory = (
        PLANTDOC_DIR /
        "train" /
        category
    )

    if directory.exists():

        images = get_images(directory)

        plantdoc_train["diseased"].extend(
            images
        )


# Test images

for category in HEALTHY_CATEGORIES:

    directory = (
        PLANTDOC_DIR /
        "test" /
        category
    )

    if directory.exists():

        images = get_images(directory)

        plantdoc_test["healthy"].extend(
            images
        )


for category in DISEASED_CATEGORIES:

    directory = (
        PLANTDOC_DIR /
        "test" /
        category
    )

    if directory.exists():

        images = get_images(directory)

        plantdoc_test["diseased"].extend(
            images
        )


print()
print("PlantDoc training:")

print(
    "Healthy:",
    len(plantdoc_train["healthy"])
)

print(
    "Diseased:",
    len(plantdoc_train["diseased"])
)


print()
print("PlantDoc test:")

print(
    "Healthy:",
    len(plantdoc_test["healthy"])
)

print(
    "Diseased:",
    len(plantdoc_test["diseased"])
)


# ============================================================
# COPY PLANTDOC
# ============================================================

print()
print("Copying PlantDoc...")


for category in [
    "healthy",
    "diseased"
]:

    copy_images(

        plantdoc_train[category],

        OUTPUT_DIR /
        "train" /
        category,

        f"plantdoc_{category}"
    )


    copy_images(

        plantdoc_test[category],

        OUTPUT_DIR /
        "test" /
        category,

        f"plantdoc_test_{category}"
    )


# ============================================================
# COPY YOUR EXISTING PLANTVILLAGE DATA
# ============================================================

print()
print("Copying existing PlantVillage dataset...")


PLANTVILLAGE_DIR = (
    PROJECT_DIR /
    "dataset"
)


# Existing dataset:
#
# dataset/train/healthy
# dataset/train/diseased
#
# dataset/val/healthy
# dataset/val/diseased
# dataset/test/healthy
# dataset/test/diseased


for split in [
    "train",
    "val",
    "test"
]:

    for category in [
        "healthy",
        "diseased"
    ]:

        source = (
            PLANTVILLAGE_DIR /
            split /
            category
        )

        destination = (
            OUTPUT_DIR /
            split /
            category
        )


        if not source.exists():

            print(
                "WARNING: missing:",
                source
            )

            continue


        images = get_images(
            source
        )


        print(
            f"PlantVillage "
            f"{split}/{category}: "
            f"{len(images)}"
        )


        copy_images(

            images,

            destination,

            f"plantvillage_{split}_{category}"
        )


# ============================================================
# CREATE VALIDATION SET
# ============================================================

print()
print("Creating validation data...")


# We already have PlantVillage validation.
#
# For PlantDoc we don't create another validation set here.
# PlantDoc test remains completely separate.


# ============================================================
# FINAL COUNTS
# ============================================================

print()
print("=" * 60)
print("COMBINED DATASET CREATED")
print("=" * 60)


for split in [
    "train",
    "val",
    "test"
]:

    print()
    print(split.upper())

    for category in [
        "healthy",
        "diseased"
    ]:

        directory = (
            OUTPUT_DIR /
            split /
            category
        )

        count = len(
            get_images(directory)
        )

        print(
            f"{category.capitalize():10s}: "
            f"{count}"
        )


print()
print("Dataset location:")

print(
    OUTPUT_DIR
)

print()
print("Done!")