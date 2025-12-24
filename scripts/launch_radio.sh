#!/bin/bash
# Enable USB Serial permissions
sudo chmod 666 /dev/ttyUSB0

# Maximize Jetson Performance (Max Clock Speed)
sudo jetson_clocks

# Navigate to code and run
cd /home/falcon/FALCON_SIH2025/src/server
python3 fal_server_v2.py