import cv2
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import threading
import time
import socket
import json
import zlib
import struct
import math
import os

# =========================
# ⚙️ CONFIGURATION
# =========================
SERVER_PORT = 9999
MODEL_STD_PATH = "/falcon_project/models/pretrained.engine"  # Standard Model
MODEL_CROWD_PATH = "/falcon_project/models/trained.engine"   # Crowd Model

# Tuning
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45       # Overlap threshold
GRID_ROWS = 5
GRID_COLS = 5

# Risk Thresholds
THRESH_DENSITY_CRITICAL = 10   # People per grid cell
THRESH_SPEED_STAMPEDE = 1.5    # Pixels movement average
THRESH_DIRECTION_ENTROPY = 1.5 # Chaos level

# =========================
# 🧠 TENSORRT WRAPPER
# =========================
class TRT_Engine:
    def __init__(self, engine_path, shared_context):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.shared_ctx = shared_context
        self.stream = cuda.Stream()
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
            
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

    def infer(self, image, target_width=512, target_height=512):
        self.shared_ctx.push()
        try:
            # 1. Preprocess
            resized = cv2.resize(image, (target_width, target_height))
            input_image = resized.transpose((2, 0, 1))[::-1] # HWC->CHW, BGR->RGB
            input_image = np.ascontiguousarray(input_image, dtype=np.float32) / 255.0
            
            np.copyto(self.inputs[0]['host'], input_image.ravel())
            
            # 2. Inference
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            # 3. Parse Output
            output = self.outputs[0]['host']
            
            # Auto-detect channels
            num_anchors = 5376 
            if output.size % num_anchors != 0:
                 # Fallback logic if shape mismatch
                 if output.size % 84 == 0:
                     num_channels = 84
                     num_anchors = int(output.size / 84)
                 elif output.size % 5 == 0:
                     num_channels = 5
                     num_anchors = int(output.size / 5)
                 else:
                     num_anchors = 8400
                     num_channels = int(output.size / num_anchors)
            else:
                 num_channels = int(output.size / num_anchors)

            predictions = output.reshape(1, num_channels, num_anchors)[0].T

            # Smart Channel Selection
            if num_channels > 5:
                scores = predictions[:, 4] # Standard Model
            else:
                scores = predictions[:, 4] # Custom Model
            
            mask = scores > CONF_THRESHOLD
            filtered = predictions[mask]
            
            if len(filtered) == 0: return np.empty((0, 4)), np.empty((0,))

            # Extract Boxes
            h_orig, w_orig = image.shape[:2]
            scale_x = w_orig / target_width
            scale_y = h_orig / target_height
            
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            
            x1 = (cx - w/2) * scale_x
            y1 = (cy - h/2) * scale_y
            x2 = (cx + w/2) * scale_x
            y2 = (cy + h/2) * scale_y
            
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            final_scores = filtered[:, 4]
            
            return boxes, final_scores

        finally:
            self.shared_ctx.pop()

# =========================
# 🦅 MAIN SERVER
# =========================
class FalconServer:
    def __init__(self):
        self.running = True
        
        # GPU Init
        cuda.init()
        self.device = cuda.Device(0)
        self.ctx = self.device.make_context()
        
        print("[INFO] Loading Models...")
        try:
            self.model_std = TRT_Engine(MODEL_STD_PATH, self.ctx)
            self.model_crowd = TRT_Engine(MODEL_CROWD_PATH, self.ctx)
            print("[SUCCESS] Models Loaded.")
        except Exception as e:
            self.ctx.pop()
            raise e
        self.ctx.pop() # Release for threads

        # Camera (GStreamer for CSI)
        gst_str = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )
        self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
             print("❌ Camera Failed to Open! Check connection.")

        # Network
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', SERVER_PORT))
        self.sock.listen(1)
        
        self.client_conn = None
        self.prev_centers = []
        self.sustain_counter = 0

    def handle_client(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
                print(f"[NET] Connected: {addr}")
                self.client_conn = conn
                while self.client_conn: time.sleep(1)
            except: self.client_conn = None

    def calculate_entropy(self, vectors):
        if len(vectors) == 0: return 0.0
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        hist, _ = np.histogram(angles, bins=8, range=(-math.pi, math.pi), density=True)
        hist += 1e-9
        p = hist / np.sum(hist)
        return -np.sum(p * np.log(p))

    def run(self):
        threading.Thread(target=self.handle_client, daemon=True).start()
        
        processed = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret: 
                time.sleep(0.1)
                continue
            
            # --- 1. INFERENCE ---
            boxes1, scores1 = self.model_std.infer(frame)
            boxes2, scores2 = self.model_crowd.infer(frame)
            
            # Merge Results
            if len(boxes1) > 0 and len(boxes2) > 0:
                all_boxes = np.concatenate((boxes1, boxes2))
                all_scores = np.concatenate((scores1, scores2))
            elif len(boxes1) > 0:
                all_boxes, all_scores = boxes1, scores1
            else:
                all_boxes, all_scores = boxes2, scores2

            # --- 2. GLOBAL NMS ---
            final_boxes = []
            centers = []
            
            if len(all_boxes) > 0:
                # Convert xyxy -> xywh for NMS
                w_arr = all_boxes[:, 2] - all_boxes[:, 0]
                h_arr = all_boxes[:, 3] - all_boxes[:, 1]
                boxes_wh = np.stack([all_boxes[:, 0], all_boxes[:, 1], w_arr, h_arr], axis=1)
                
                indices = cv2.dnn.NMSBoxes(boxes_wh.tolist(), all_scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
                
                if len(indices) > 0:
                    indices = indices.flatten()
                    final_boxes = all_boxes[indices]
                    
                    cx = (final_boxes[:, 0] + final_boxes[:, 2]) / 2
                    cy = (final_boxes[:, 1] + final_boxes[:, 3]) / 2
                    centers = np.column_stack((cx, cy))
                    
                    for b in final_boxes:
                        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

            # --- 3. RISK PHYSICS ---
            head_count = len(centers)
            avg_motion = 0.0
            entropy = 0.0
            
            if len(self.prev_centers) > 0 and len(centers) > 0:
                prev = np.array(self.prev_centers)
                curr = np.array(centers)
                dists = np.linalg.norm(curr[:, None, :] - prev[None, :, :], axis=2)
                min_dists = np.min(dists, axis=1)
                match_indices = np.argmin(dists, axis=1)
                
                valid = min_dists < 50
                if np.any(valid):
                    matched_curr = curr[valid]
                    matched_prev = prev[match_indices[valid]]
                    vectors = matched_curr - matched_prev
                    norms = np.linalg.norm(vectors, axis=1)
                    avg_motion = np.mean(norms)
                    entropy = self.calculate_entropy(vectors)

            self.prev_centers = centers

            # --- 4. RISK STATE MACHINE ---
            active_conditions = 0
            if head_count > 20: active_conditions += 1
            if avg_motion > THRESH_SPEED_STAMPEDE and head_count > 10: active_conditions += 1
            if entropy > THRESH_DIRECTION_ENTROPY and head_count > 10: active_conditions += 1

            if active_conditions >= 2:
                self.sustain_counter += 1
            else:
                self.sustain_counter = max(0, self.sustain_counter - 1)

            risk = "LOW"
            if self.sustain_counter > 10: risk = "HIGH"
            elif active_conditions >= 1: risk = "MEDIUM"

            # --- 5. SEND DATA (Throttled) ---
            if self.client_conn and processed % 2 == 0:
                try:
                    payload = {
                        "frame": processed,
                        "heads": int(head_count),
                        "max_cell": 0,
                        "motion": float(avg_motion),
                        "entropy": float(entropy),
                        "risk": risk
                    }
                    j_bytes = zlib.compress(json.dumps(payload).encode('utf-8'))
                    self.client_conn.sendall(struct.pack("!L", len(j_bytes)) + j_bytes)
                    _, i_bytes = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    self.client_conn.sendall(struct.pack("!L", len(i_bytes)) + i_bytes.tobytes())
                except:
                    self.client_conn.close()
                    self.client_conn = None

            processed += 1
            if processed % 15 == 0:
                print(f"[STATUS] Count: {head_count} | Motion: {avg_motion:.2f} | Risk: {risk}")

if __name__ == "__main__":
    FalconServer().run()