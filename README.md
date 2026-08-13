# 🌱 Plant Health Check AI

A computer vision AI project that uses a ResNet-18 neural network to detect whether a plant leaf appears healthy or potentially diseased.

This project runs on an NVIDIA Jetson Orin Nano using jetson-containers, PyTorch, CUDA, and OpenCV.

## 🎯 Project Goal

The goal of this project is to create an AI system that helps home gardeners quickly identify possible plant health problems from leaf images.

The AI classifies plant leaves into two categories:

- 🟢 Healthy
- 🔴 Diseased

The model looks for visible signs such as:

- 🟤 Brown or dark spots
- 🟡 Yellow discoloration
- 🕳️ Holes
- 🦠 Mold-like areas
- 🍂 Rotting areas
- 🥀 Dry or withered leaf texture

A healthy leaf generally has an even green color and a fresh appearance.

> ⚠️ Note: This is a two-class image classifier. It does not identify the exact disease and may not detect very small early-stage symptoms.

## 📸 Project Demo

### 🖼️ Project Screenshots

Here are screenshots showing the Plant Health Check AI running on the NVIDIA Jetson Orin Nano.

![Plant Health AI Screenshot 1](screenshots/plant-health-1.png)

![Plant Health AI Screenshot 2](screenshots/plant-health-2.png)

![Plant Health AI Screenshot 3](screenshots/plant-health-3.png)

### 🎥 Project Demo Video

[▶️ Watch the Plant Health AI Demo Video](https://drive.google.com/file/d/1NfCn8E1BQCwIiFXmUJelmhrzyfaomK9H/view?usp=sharing)

---

## 🤖 AI Model

This project uses ResNet-18 with transfer learning.

A pretrained ResNet-18 model was fine-tuned using plant leaf images from:

- 🌿 PlantVillage Dataset
- 🌱 PlantDoc Dataset

The final model classifies images into:

```text
healthy
diseased
```

## 🖥️ Hardware

- 🧠 NVIDIA Jetson Orin Nano
- 📷 USB Webcam
- ⚡ NVIDIA GPU

## 💻 Software

- 🐍 Python 3.10
- 🔥 PyTorch 2.2
- 🖼️ Torchvision 0.17
- 📷 OpenCV 4.8
- 🚀 CUDA 12.2
- 🧩 NVIDIA JetPack 6.0
- 📦 jetson-containers
- 🧠 ResNet-18

## 📊 Dataset

This project combines the PlantVillage and PlantDoc datasets.

### Combined Dataset

| Dataset Split | Healthy | Diseased | Total |
|---|---:|---:|---:|
| 🏋️ Training | 2,360 | 3,180 | 5,540 |
| 🔎 Validation | 200 | 200 | 400 |
| 🧪 Testing | 290 | 346 | 636 |

The datasets are not included in this GitHub repository because of their size and dataset licensing and usage requirements.

## 📈 Model Performance

The final ResNet-18 model achieved:

- 🎯 Test Accuracy: **97.64%**
- 🟢 Healthy Recall: **99.31%**
- 🔴 Diseased Recall: **96.24%**

### Confusion Matrix

| Actual / Predicted | Diseased | Healthy |
|---|---:|---:|
| 🔴 Diseased | 333 | 13 |
| 🟢 Healthy | 2 | 288 |

## 📁 Project Structure

```text
plant-health-ai/
│
├── dataset/
│   ├── raw/
│   │   ├── PlantVillage-Dataset/
│   │   └── PlantDoc-Dataset/
│   │
│   └── combined/
│       ├── train/
│       │   ├── healthy/
│       │   └── diseased/
│       ├── val/
│       │   ├── healthy/
│       │   └── diseased/
│       └── test/
│           ├── healthy/
│           └── diseased/
│
├── models/
│   └── resnet18_plant_combined.pth
│
├── scripts/
│   ├── prepare_dataset.py
│   └── prepare_combined_dataset.py
│
├── src/
│   ├── train.py
│   ├── train_combined.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── predict_combined.py
│   └── webcam.py
│
└── README.md
```

## 📦 Running with jetson-containers

Start the PyTorch container:

```bash
cd ~/plant-health-ai

jetson-containers run \
--volume ~/plant-health-ai:/workspace/plant-health-ai \
$(autotag l4t-pytorch)
```

Inside the container:

```bash
cd /workspace/plant-health-ai
```

## ⚡ Verify GPU

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected result:

```text
CUDA: True
GPU: Orin
```

## 🏋️ Training

Train the combined PlantVillage + PlantDoc model:

```bash
python3 src/train_combined.py
```

The trained model is saved to:

```text
models/resnet18_plant_combined.pth
```

## 🔎 Evaluate the Model

```bash
python3 src/evaluate.py
```

Example:

```text
Accuracy: 97.64%

Diseased Recall: 96.24%
Healthy Recall: 99.31%
```

## 🖼️ Test an Image

```bash
python3 src/predict_combined.py /workspace/plant-health-ai/my_leaf.jpg
```

Example healthy result:

```text
Prediction: HEALTHY
Confidence: 99.98%

Diseased  : 0.02%
Healthy   : 99.98%

# OK: Leaf appears healthy.
```

Example diseased result:

```text
Prediction: DISEASED
Confidence: 99.47%

Diseased  : 99.47%
Healthy   : 0.53%

# WARNING: Possible unhealthy leaf detected.
```

## 📷 Live Webcam Detection

The project also supports real-time plant health detection using a webcam.

Run:

```bash
python3 src/webcam.py
```

Pipeline:

```text
📷 Camera
   ↓
🖼️ OpenCV
   ↓
⚙️ Image Preprocessing
   ↓
🧠 ResNet-18
   ↓
🌱 Healthy / Diseased
   ↓
📊 Confidence Score
   ↓
⚠️ Warning
```

The webcam program displays:

- 📷 Camera image
- 🔎 Prediction
- 📊 Confidence
- 🟢 Healthy/Diseased status
- ⚠️ Warning when a possible unhealthy leaf is detected

## 🎥 Checking the Webcam

Check available camera devices:

```bash
ls -l /dev/video*
```

Example:

```text
/dev/video0
/dev/video1
```

Test the webcam:

```bash
python3 -c "import cv2; cap=cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2); print('Camera opened:', cap.isOpened()); cap.release()"
```

Expected result:

```text
Camera opened: True
```

## 🚨 Disease Warning

Example threshold:

```text
DISEASE_THRESHOLD = 0.70
```

Detection logic:

```text
Diseased probability < 70%
    ↓
🟢 Leaf appears healthy

Diseased probability >= 70%
    ↓
⚠️ WARNING: Possible unhealthy leaf detected
```

Examples:

```text
18% diseased → 🟢 Healthy
45% diseased → 🟢 Healthy
65% diseased → 🟢 Healthy
72% diseased → ⚠️ Warning
99% diseased → ⚠️ Warning
```

## 🌿 Plant Health Detection Logic

### 🟢 Healthy Leaf

The system should normally classify a leaf as healthy when it has:

- 🟢 Smooth green color
- 🟢 Even coloration
- ❌ No obvious brown spots
- ❌ No large yellow areas
- ❌ No visible holes
- ❌ No visible mold
- ❌ No obvious rotting
- 🌱 Fresh-looking leaf texture

Output:

```text
OK: Leaf appears healthy.
```

### 🔴 Potentially Diseased Leaf

The system may issue a warning when it detects:

- 🟤 Brown spots
- ⚫ Dark spots
- 🟡 Large yellow discoloration
- 🕳️ Holes
- 🦠 Mold-like areas
- 🍂 Rotting areas
- 🥀 Dry or heavily withered texture

Output:

```text
WARNING: Possible unhealthy leaf detected.
```

## 🧪 Testing Results

Example:

```text
my_leaf.jpg

Prediction: HEALTHY
Confidence: 99.98%
```

Example:

```text
my_leaf1.jpg

Prediction: DISEASED
Confidence: 99.47%
```

Example:

```text
my_leaf2.jpg

Prediction: HEALTHY
Confidence: 81.08%
```

Example:

```text
my_leaf3.jpg

Prediction: HEALTHY
Confidence: 100.00%
```

## 🛠️ Technologies Used

- 🐍 Python
- 🔥 PyTorch
- 🖼️ Torchvision
- 🧠 ResNet-18
- 📷 OpenCV
- ⚡ CUDA
- 🧩 NVIDIA JetPack
- 🖥️ NVIDIA Jetson Orin Nano
- 📦 jetson-containers
- 🌿 PlantVillage
- 🌱 PlantDoc

## 🚀 Future Improvements

- 📸 More real-world plant images
- 🏠 More houseplant images
- 🌿 More plant species
- 🦠 More disease categories
- 🔬 Exact disease identification
- 🌱 Plant species identification
- 📷 Better webcam performance
- ⚡ TensorRT optimization
- 🚀 Faster real-time inference
- 📥 Automatic image collection
- 💡 Improved lighting handling
- ✂️ Background removal
- 🌐 Web interface
- 📱 Mobile application
- 📊 Automatic disease history tracking

## ⚠️ Limitations

This project is an educational computer vision system and is not a professional plant disease diagnosis system.

The model may:

- ❌ Make incorrect predictions
- 🔍 Miss very small early-stage symptoms
- 💡 Perform differently under different lighting conditions
- 📐 Perform differently with unusual camera angles
- 🌿 Perform differently on plant species not well represented in the training data
- 🖼️ Produce incorrect results on heavily cluttered backgrounds

The confidence score represents the model's prediction and is not a guarantee of plant health.

## 🎯 Future Goal

The final goal is a real-time system where a user points a webcam at a plant leaf and the Jetson Orin Nano automatically analyzes it.

```text
🌿 Plant Leaf
      ↓
📷 Webcam
      ↓
⚙️ Preprocessing
      ↓
🧠 ResNet-18
      ↓
   ┌───────────────┐
   ↓               ↓
🟢 Healthy      🔴 Diseased
   ↓               ↓
✅ OK            ⚠️ Warning
```

## 📋 Project Information

**Project:** 🌱 Plant Health Check AI

**Platform:** 🖥️ NVIDIA Jetson Orin Nano

**Model:** 🧠 ResNet-18

**Framework:** 🔥 PyTorch

**Computer Vision:** 📷 OpenCV

**Container:** 📦 jetson-containers

**Dataset:** 🌿 PlantVillage + PlantDoc

**Classes:** 🟢 Healthy / 🔴 Diseased

**Test Accuracy:** 🎯 97.64%

## 📜 License

This project is intended for educational and research purposes.

The PlantVillage and PlantDoc datasets are separate resources and are subject to their respective licenses and usage requirements.

## 🎉 Conclusion

Plant Health Check AI demonstrates how a deep learning model can be deployed on an edge AI device such as the NVIDIA Jetson Orin Nano.

The project combines:

```text
📊 Dataset
   ↓
🏋️ Training
   ↓
🧠 ResNet-18
   ↓
📈 Evaluation
   ↓
🖼️ Image Prediction
   ↓
📷 Webcam Detection
```

The system provides a simple way to detect possible plant health problems using computer vision and real-time AI inference.

---

🌱 **Plant Health Check AI**  
🤖 **ResNet-18 + PyTorch**  
🖥️ **NVIDIA Jetson Orin Nano**  
📷 **Real-Time Computer Vision**
