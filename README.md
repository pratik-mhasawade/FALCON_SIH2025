# 🦅 FALCON V2: Edge-AI Stampede Prevention System

## 📂 Project Structure
* `src/server/`: Code running on the **Drone** (Jetson Nano).
* `src/client/`: Dashboard running on the **Ground Station** (Laptop).
* `assets/models/`: AI Engine files (must be placed here).
* `scripts/`: Auto-start and utility scripts.

## 🚀 Quick Start Guide

### 1. Drone Setup (Server)
1. Copy `src/server` and `assets` to the Jetson Nano.
2. Install dependencies: `pip3 install -r src/server/requirements_server.txt`
3. Run the server: 
   ```bash
   python3 src/server/fal_server_v2.py


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Above is newer one//////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 🦅 Edge-AI Drone for Proactive Stampede Prevention (Team FALCON)

### Problem Statement (SIH25118)
Large gatherings are prone to deadly stampedes due to delayed response times and lack of real-time monitoring. Current CCTV solutions have blind spots, and manual drones rely on human vigilance.

### Solution
An autonomous AI-powered drone that monitors crowd density and panic motion in real-time using an onboard NVIDIA Jetson Nano. It operates independently of cloud connectivity and uses long-range radio telemetry to alert ground authorities instantly.

---

### 🚀 Key Features
* **Onboard Edge AI:** Runs YOLOv8 and Optical Flow on Jetson Nano (No Internet needed).
* **Long-Range Telemetry:** Transmits alerts and visual proof over 433MHz Radio (1km+ range).
* **Proactive Alerts:** Detects "Rate of Rise" in crowd density before it becomes critical.
* **Active Deterrence:** Drone can trigger Sirens/Lights to disperse crowds.
* **Ground Dashboard:** Real-time tactical map, trend graphs, and SOS reporting.

### 🛠️ Hardware Stack
* **Compute:** NVIDIA Jetson Nano (4GB)
* **Vision:** Raspberry Pi Camera V2 (CSI)
* **Flight:** Pixhawk 2.4.8 + F450 Frame
* **Comms:** Holybro SiK Telemetry Radio V3 (433MHz)

### 💻 Software Stack
* **Language:** Python 3.8
* **AI:** TensorRT, PyCUDA, OpenCV
* **GUI:** Tkinter, Matplotlib
* **Comms:** PySerial (Custom Packet Protocol)

---

### ⚙️ Installation & Setup

#### 1. Jetson Nano (Server)
1.  Connect Camera and SiK Radio.
2.  Install dependencies:
    ```bash
    sudo usermod -a -G dialout $USER
    sudo reboot
    pip3 install -r requirements_server.txt
    ```
3.  Run the launcher:
    ```bash
    ./launch_radio.sh
    ```

#### 2. Laptop (Ground Station)
1.  Connect the Ground Radio via USB.
2.  Install dependencies:
    ```bash
    pip install -r requirements_client.txt
    ```
3.  Run the Dashboard:
    ```bash
    python falcon_radio_client.py
    ```

---

### 📊 Usage
1.  **Standby:** System boots into low-power mode.
2.  **Start:** Click **"START MISSION"** on the dashboard.
3.  **Monitor:** Watch the "Density Trend" graph.
4.  **Alert:** If Risk hits **HIGH**, the dashboard screams and turns red.
5.  **Action:** Click **"SOS REPORT"** to save a snapshot for the police.