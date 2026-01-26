# 🚦 VR-TSD: A Real-World Dataset and a Lightweight YOLOv8n Model for Traffic Sign Recognition in Vietnam

**Paper:** VR-TSD: A Real-World Dataset and a Lightweight YOLOv8n Model for Traffic Sign Recognition in Vietnam  
**Conference:** CSA 2025 – The 17th International Conference on Computer Science and its Applications

---
**Demo:** Integrate voice notification feature: https://drive.google.com/file/d/1JtekjwLvfJC0tHHrWJdlvd40RyeMrAkl/view
## 1. Introduction

Accurate and real-time traffic sign recognition is a key component of Advanced Driver Assistance Systems (ADAS) and autonomous vehicles. This task requires not only high recognition accuracy but also fast and stable inference, especially when deployed on resource-constrained edge devices.

Most existing studies rely on foreign datasets or limited domestic datasets that do not fully reflect real traffic conditions in Vietnam. In practice, traffic signs may appear faded, partially occluded, or captured under adverse lighting and weather conditions, leading to degraded model performance when deployed in real-world scenarios.

To address this gap, this paper proposes a new real-world traffic sign dataset and a lightweight detection framework tailored to Vietnamese traffic environments.

---

## 2. VR-TSD Dataset

The **Vietnamese Real-world Traffic Sign Dataset (VR-TSD)** is constructed from real traffic scenes captured in Vietnam with the following characteristics:

- 58 common Vietnamese traffic sign categories
- Diverse and challenging acquisition conditions:
  - Low-light and nighttime scenes
  - Rainy and adverse weather conditions
  - Occluded and faded traffic signs
  - Dense urban traffic and national highways

The dataset is designed to bridge the gap between academic benchmarks and real-world deployment scenarios.

---

## 3. Proposed Method

Based on the VR-TSD dataset, we develop a lightweight traffic sign recognition system using the **YOLOv8n** architecture. The model is optimized to achieve a balanced trade-off among:

- Detection accuracy
- Real-time inference speed
- Compact model size for edge deployment

The proposed pipeline covers data preprocessing, model training, evaluation, and deployment on mobile devices.

---

## 4. Experimental Results

The YOLOv8n-based model achieves strong performance on the VR-TSD dataset:

- **mAP@0.5:** 0.9434  
- **mAP@0.5:0.95:** 0.7567  

**Inference speed:**
- 168.86 FPS on NVIDIA A100 GPU
- 77.5 FPS on iPhone 11 Pro Max

**Model size:** approximately 6 MB

The results demonstrate that the proposed system satisfies real-time requirements for ADAS applications, where traffic sign recognition typically requires a minimum of 10 FPS.

---

## 5. Contributions

The main contributions of this work are summarized as follows:

1. A new public real-world Vietnamese traffic sign dataset with 58 categories collected under challenging conditions.
2. A lightweight YOLOv8n-based detection model optimized for edge devices with high accuracy and real-time performance.
3. An end-to-end validated system tested on mainstream mobile devices, demonstrating practical feasibility for ADAS deployment.

---

## 6. Potential Applications

- Advanced Driver Assistance Systems (ADAS)
- Low- to mid-level autonomous driving systems
- Intelligent traffic monitoring
- Edge AI applications on mobile and embedded devices
