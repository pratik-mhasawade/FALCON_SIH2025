#!/bin/bash

# --- 1. HARDWARE SETUP ---
# Give permission to the USB Radio
if [ -e /dev/ttyUSB0 ]; then
    echo "[SETUP] Radio Found at ttyUSB0"
    sudo chmod 666 /dev/ttyUSB0
else
    echo "[SETUP] No Radio Found! Checking ttyUSB1..."
    if [ -e /dev/ttyUSB1 ]; then
        sudo chmod 666 /dev/ttyUSB1
    fi
fi

# Maximize Jetson Performance (Prevents lag)
# This forces the GPU/CPU to max frequency
sudo jetson_clocks

# --- 2. PATH SETUP ---
# Ensure we are in the correct directory so relative paths (../../assets) work
cd /home/falcon/FALCON_SIH2025/src/server

# --- 3. LAUNCH SERVER ---
# We run it with python3
echo "[LAUNCH] Starting FALCON Server..."
python3 fal_server_v2.py