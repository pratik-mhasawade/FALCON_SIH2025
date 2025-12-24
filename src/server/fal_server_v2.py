import cv2
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import threading
import time
import serial
import json
import struct
import math
import os
from collections import deque

# --- CONFIGURATION ---
# Hardware Paths
SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 57600
# Update these paths relative to where you run the script
MODEL_STD_PATH = "../../assets/models/pretrained.engine"
MODEL_CROWD_PATH = "../../assets/models/trained.engine"

# Command Protocol (Hex Codes)
CMD_PING = 0x10
CMD_START = 0x11
CMD_STOP = 0x12
CMD_SHUTDOWN = 0x14
CMD_REQ_IMG = 0x05

# System Parameters
CONF_THRESHOLD = 0.25      # AI Confidence requirement
FRAME_SKIP = 3             # Process every 3rd frame (approx 10 FPS processing)
CHUNK_SIZE = 64            # Bytes per image packet (Critical for radio stability)
IMAGE_INTERVAL = 5.0       # Auto-send image every 5 seconds
GRID_ROWS = 5              # Density Map Rows
GRID_COLS = 5              # Density Map Cols

class SerialManager:
    """Handles thread-safe Radio communication"""
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.lock = threading.Lock()
        self.connect()

    def connect(self):
        while self.ser is None:
            try:
                print(f"[RADIO] Connecting to {self.port}...")
                self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
                print(f"[RADIO] [OK] Connected.")
            except: time.sleep(2)

    def send_packet(self, type_byte, payload):
        if not self.ser: return
        try:
            length = len(payload)
            # Protocol: [HEADER 0xFACE][TYPE][LEN][PAYLOAD][CHECKSUM]
            header = struct.pack('!HBH', 0xFACE, type_byte, length)
            checksum = 0
            for b in payload: checksum ^= b
            with self.lock:
                self.ser.write(header + payload + struct.pack('B', checksum))
        except:
            self.ser = None
            threading.Thread(target=self.connect, daemon=True).start()

    def read_command(self):
        if not self.ser or self.ser.in_waiting == 0: return None
        try:
            raw = self.ser.read(self.ser.in_waiting)
            if bytes([CMD_START]) in raw: return "START"
            if bytes([CMD_STOP]) in raw: return "STOP"
            if bytes([CMD_SHUTDOWN]) in raw: return "SHUTDOWN"
            if bytes([CMD_REQ_IMG]) in raw: return "IMG"
            if bytes([CMD_PING]) in raw: return "PING"
        except: pass
        return None

class TRT_Engine:
    """Wrapper for NVIDIA TensorRT Models"""
    def __init__(self, engine_path, shared_context):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.shared_ctx = shared_context
        self.stream = cuda.Stream()
        if not os.path.exists(engine_path): raise FileNotFoundError(f"Missing: {engine_path}")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer_fast(self, input_data, original_shape):
        self.shared_ctx.push()
        try:
            np.copyto(self.inputs[0]['host'], input_data)
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            output = self.outputs[0]['host']
            # YOLOv8 Output Parsing
            num_anchors = 8400 if output.size % 8400 == 0 else 5376
            num_channels = int(output.size / num_anchors)
            predictions = output.reshape(1, num_channels, num_anchors)[0].T
            scores = predictions[:, 4]
            mask = scores > CONF_THRESHOLD
            filtered = predictions[mask]
            
            if len(filtered) == 0: return np.empty((0, 4)), np.empty((0,))
            
            # Box Scaling
            h, w = original_shape[:2]
            sx, sy = w / 512, h / 512
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            x1 = (cx - w/2) * sx
            y1 = (cy - h/2) * sy
            x2 = (cx + w/2) * sx
            y2 = (cy + h/2) * sy
            return np.stack([x1, y1, x2, y2], axis=1), filtered[:, 4]
        finally:
            self.shared_ctx.pop()

class FalconRadioServer:
    def __init__(self):
        self.running = True
        self.state = "STANDBY"
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        self.radio = SerialManager(SERIAL_PORT, BAUD_RATE)
        self.model_std = None
        self.model_crowd = None
        
        # GStreamer Pipeline for CSI Camera (Raspberry Pi Cam V2)
        gst = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        self.cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
        
        self.image_chunks = deque()
        self.sustain_counter = 0
        self.last_img_time = 0

    def load_models(self):
        if self.model_std: return
        self.send_status("LOADING AI MODELS...")
        try:
            self.model_std = TRT_Engine(MODEL_STD_PATH, self.ctx)
            self.model_crowd = TRT_Engine(MODEL_CROWD_PATH, self.ctx)
            self.send_status("AI LOADED. READY.")
        except Exception as e:
            self.send_status(f"AI ERROR: {str(e)[:20]}")

    def send_status(self, text):
        payload = json.dumps({"s": text}).encode('utf-8')
        self.radio.send_packet(0x03, payload)

    def prepare_chunks(self, frame):
        # Resize to 160x120 thumbnail for radio transmission
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (160, 120))
        _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 30])
        data = buf.tobytes()
        total = math.ceil(len(data) / CHUNK_SIZE)
        return [struct.pack('BB', i, total) + data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] for i in range(total)]

    def run(self):
        processed = 0
        print("[SYS] System Started.")
        
        while self.running:
            # 1. READ COMMANDS
            cmd = self.radio.read_command()
            if cmd == "START":
                self.load_models()
                self.state = "ACTIVE"
                self.send_status("MISSION STARTED")
            elif cmd == "STOP":
                self.state = "STANDBY"
                self.send_status("MISSION PAUSED")
            elif cmd == "SHUTDOWN":
                self.send_status("SHUTTING DOWN")
                time.sleep(1)
                os.system("shutdown now")
            elif cmd == "PING":
                self.send_status(f"PONG: {self.state}")

            # 2. STANDBY LOOP
            if self.state == "STANDBY":
                ret, _ = self.cap.read()
                time.sleep(0.1)
                continue

            # 3. ACTIVE LOOP
            ret, frame = self.cap.read()
            if not ret: continue

            if processed % FRAME_SKIP == 0 and self.model_std:
                # Pre-processing
                resized = cv2.resize(frame, (512, 512))
                inp = np.ascontiguousarray(resized.transpose((2, 0, 1))[::-1], dtype=np.float32) / 255.0
                flat = inp.ravel()
                
                # Inference
                b1, _ = self.model_std.infer_fast(flat, frame.shape)
                b2, _ = self.model_crowd.infer_fast(flat, frame.shape)
                
                # Simple logic for now: combine counts
                heads = len(b1) + len(b2)
                
                # Risk Logic [cite: 55, 56, 57]
                risk = "L"
                if heads > 55: risk = "M"
                if heads > 110: risk = "H"
                
                # Send Telemetry
                if processed % 3 == 0:
                    payload = json.dumps({"h": heads, "r": risk, "f": processed}).encode('utf-8')
                    self.radio.send_packet(0x01, payload)

            # 4. IMAGE TRANSMISSION
            now = time.time()
            if (cmd == "IMG") or (now - self.last_img_time > IMAGE_INTERVAL and not self.image_chunks):
                self.image_chunks = deque(self.prepare_chunks(frame))
                self.last_img_time = now

            if self.image_chunks:
                self.radio.send_packet(0x02, self.image_chunks.popleft())
                time.sleep(0.015) # Pacing delay
            
            processed += 1

if __name__ == "__main__":
    FalconRadioServer().run()