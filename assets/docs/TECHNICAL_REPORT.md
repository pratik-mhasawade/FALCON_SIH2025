This is a comprehensive **Technical Report and Developer Guide**. You should save this content as **`docs/TECHNICAL_REPORT.md`** in your project folder.

It covers every aspect of the system: from the high-level concept to the gritty engineering challenges (YOLO versions, Radio bottlenecks, CPU heat) and how you solved them. This is exactly what judges and future developers need to see.

---

# 🦅 FALCON V2: Technical Report & Developer Guide

**Project:** Edge-AI Drone for Proactive Stampede Prevention
**Team:** FALCON (ID: 52880)
**Problem Statement:** SIH25118

---

## 1. Project Overview

### 1.1 The Core Mission

Large gatherings (like Kumbh Mela) suffer from a "Blind Spot" problem. CCTV is fixed, and 4G/5G networks collapse when millions of people gather.

**FALCON** is an autonomous drone system that acts as an "Eye in the Sky." It processes video **onboard** (Edge AI) to detect crowd density and panic motion, then transmits alerts via a private **Long-Range Radio Link** (433MHz), completely bypassing the need for internet/cloud connectivity.

### 1.2 Technical Stack

* 
**Edge Compute:** NVIDIA Jetson Nano (4GB) running JetPack 4.6.


* 
**AI Engine:** TensorRT (Optimized YOLOv8) + OpenCV (Optical Flow).


* 
**Telemetry:** Holybro SiK Radio V3 (433 MHz) at 57600 Baud.


* 
**Flight Controller:** Pixhawk 2.4.8.


* 
**Power:** Dual-Rail System (LiPo for Motors, Buck Converter for Jetson).



---

## 2. Parameter & Extensibility Guide

This system is built to be tuned. The following parameters in `src/server/fal_server_v2.py` control the system's behavior.

### 2.1 Critical Tuning Parameters

| Parameter | Default | Description & Tuning Guide |
| --- | --- | --- |
| **`FRAME_SKIP`** | `3` | **What it does:** Processes only every 3rd frame (e.g., Frame 0, 3, 6...).<br>

<br>**Impact:** Increasing this (e.g., to 5) lowers CPU usage and heat but makes motion detection "jerky." Decreasing to 1 maximizes accuracy but may overheat the Jetson Nano. |
| **`CHUNK_SIZE`** | `64` | **What it does:** Splits image data into small packets for radio transmission.<br>

<br>**Impact:** <br>

<br>• **64-128:** Stable. Best for noisy environments.<br>

<br>• **>256:** Faster image load, but high risk of packet corruption/loss. |
| **`CONF_THRESHOLD`** | `0.25` | **What it does:** Ignores detections with low AI confidence.<br>

<br>**Impact:** Increase to `0.4` if you see "ghosts" (false positives). Decrease to `0.15` if the drone is flying very high and missing people. |
| **`IMAGE_INTERVAL`** | `5.0` | **What it does:** Automatically sends a thumbnail every 5 seconds.<br>

<br>**Impact:** Increase this to free up bandwidth for Telemetry (JSON) updates. |

### 2.2 Extensibility (How to Upgrade)

#### **A. Changing the Camera (CSI to USB)**

Currently, the system uses a GStreamer pipeline for the Raspberry Pi NoIR Camera (CSI). To use a USB Webcam:

1. Open `fal_server_v2.py`.
2. Locate `self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)`.
3. Replace it with: `self.cap = cv2.VideoCapture(0)` (0 is the USB device index).

#### **B. Swapping AI Models**

If you train a more accurate model (e.g., YOLOv8-Medium) or a specific model for night vision:

1. **Export:** Convert your PyTorch model to ONNX.
2. **Optimize:** Run `trtexec --onnx=model.onnx --saveEngine=new_model.engine` on the Jetson.
3. **Deploy:** Place `new_model.engine` in `assets/models/` and update `MODEL_CROWD_PATH` in the server script.

---

## 3. Performance Metrics

* **Inference Speed:** ~18-22 FPS (using TensorRT FP16 optimization on Jetson Nano).
* 
**Radio Range:** ~1 km Line-of-Sight (using 433MHz @ 100mW).


* **Latency:**
* **Telemetry (JSON):** < 200ms (Near Real-Time).
* **Image Transmission:** ~3-5 seconds per image (due to 57600 baud bandwidth limit).


* 
**Flight Time:** ~18-22 minutes (using 3S 6800mAh Battery).


* 
**Active Coverage:** ~27.5 m² footprint at 5m altitude.



---

## 4. The Evolution: Engineering Challenges & Solutions

This project evolved through several failures. Here is the technical story of how we reached **FALCON V2**.

### **🔴 Challenge 1: The "Ultralytics" & YOLO Version Problem**

* **The Problem:** Initially, we tried running standard `ultralytics` YOLOv8 PyTorch models (`.pt` files).
* **The Failure:** The Jetson Nano's CPU spiked to 100%, and we got **1-2 FPS**. The python environment was heavy, and dependencies kept breaking on the older JetPack 4.6 OS.
* **The Solution:** We abandoned PyTorch for inference. We converted the models to **TensorRT Engines (`.engine`)**.
* *Result:* FPS jumped from 2 to **20+**.
* *Benefit:* TensorRT optimizes the neural network layers specifically for the Nano's Maxwell GPU, bypassing the CPU bottleneck.



### **🔴 Challenge 2: The CPU "Meltdown" (Thermal Throttling)**

* **The Problem:** Running AI + Video Processing continuously caused the Jetson to overheat and shut down mid-flight.
* **The Failure:** We were processing every single frame (30 FPS) which was unnecessary for crowd trends.
* **The Solution:**
1. **Frame Skipping:** We implemented `FRAME_SKIP = 3` logic. We only run heavy AI on frames 0, 3, 6...
2. **Standby Architecture:** We redesigned the code so the AI doesn't run on boot. It stays in a "Standby" loop (Low Power) until the operator clicks "START MISSION".



### **🔴 Challenge 3: The "Drinking Straw" Bandwidth (Communication)**

* **The Problem:** We needed to send Video + Data over a Telemetry Radio. These radios are designed for simple text (MAVLink), not images. They have a tiny bandwidth of **57600 bits per second**.
* **The Failure:** Trying to send a raw image froze the radio link for 20 seconds. All telemetry (risk alerts) stopped during the transfer.
* **The Solution:** The **"Chunking & Interleaving" Protocol**.
* *Step A:* We resize the image to a tiny 160x120 thumbnail.
* *Step B:* We slice the image into tiny **64-byte chunks**.
* *Step C:* We transmit chunks *in between* important JSON packets.
* *Result:* We get a live image update every ~5 seconds WITHOUT blocking the critical high-speed risk alerts.



### **🔴 Challenge 4: Headless Operation**

* **The Problem:** In the field, you can't plug a monitor/keyboard into a drone to start the script.
* **The Solution:** We created a **`systemd` service (`falcon.service`)** that auto-starts the Python script on boot. Combined with the "Remote Control" logic (Start/Stop/Shutdown commands via Radio), the laptop operator has full control over the hardware without ever needing SSH.

---

### **5. Final Architecture Verdict**

FALCON V2 is no longer just a "project code." It is a **distributed system**.

* **The Drone** is an intelligent Edge node that manages its own power and processing.
* **The Laptop** is a Tactical Command Center that receives synthesized intelligence, not just raw data.
* **The Protocol** is the custom-built bridge that makes this collaboration possible over miles of distance without internet.