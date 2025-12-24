import socket
import json
import zlib
import struct
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk
import threading
import time
import datetime

# ==========================================
# ⚙️ CONFIGURATION (EDIT THIS!)
# ==========================================
#SERVER_IP = '192.168.100.2'  # <--- REPLACE WITH JETSON IP
SERVER_IP = 'falcon.local'
SERVER_PORT = 9999
WINDOW_TITLE = "FALCON COMMAND CENTER"
# ==========================================

class FalconDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(f"{WINDOW_TITLE} - {SERVER_IP}")
        self.root.geometry("1280x720")
        self.root.configure(bg="#1e1e1e") # Dark Theme
        
        # Style Configuration
        self.setup_styles()
        
        # --- UI LAYOUT ---
        # 1. Header
        self.header = tk.Frame(root, bg="#2d2d2d", height=50)
        self.header.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_title = tk.Label(self.header, text="🦅 FALCON AI: CROWD MONITORING SYSTEM", 
                                  font=("Segoe UI", 16, "bold"), bg="#2d2d2d", fg="#ecf0f1")
        self.lbl_title.pack(side=tk.LEFT, padx=20, pady=10)
        
        self.lbl_clock = tk.Label(self.header, text="--:--:--", 
                                  font=("Consolas", 14), bg="#2d2d2d", fg="#bdc3c7")
        self.lbl_clock.pack(side=tk.RIGHT, padx=20)

        # 2. Main Content Area
        self.main_content = tk.Frame(root, bg="#1e1e1e")
        self.main_content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 2A. Video Feed (Left)
        self.video_frame = tk.Frame(self.main_content, bg="black", bd=2, relief=tk.SUNKEN)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.lbl_video = tk.Label(self.video_frame, text="WAITING FOR DRONE FEED...", 
                                  bg="black", fg="#7f8c8d", font=("Arial", 16))
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        # 2B. Data Panel (Right)
        self.data_panel = tk.Frame(self.main_content, bg="#2c3e50", width=350)
        self.data_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.data_panel.pack_propagate(False) # Force width
        
        # --- METRICS WIDGETS ---
        self.create_risk_indicator()
        self.create_metric_cards()
        
        # 3. Status Bar
        self.status_bar = tk.Label(root, text="System Idle", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#34495e", fg="white")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # --- NETWORK THREAD ---
        self.running = True
        self.last_packet_time = 0
        self.thread = threading.Thread(target=self.network_loop, daemon=True)
        self.thread.start()
        
        # Start Clock
        self.update_clock()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.font_metric = ("Segoe UI", 12)
        self.font_value = ("Consolas", 24, "bold")

    def create_risk_indicator(self):
        self.frm_risk = tk.Frame(self.data_panel, bg="#7f8c8d", pady=20)
        self.frm_risk.pack(fill=tk.X)
        
        tk.Label(self.frm_risk, text="CURRENT THREAT LEVEL", 
                 font=("Segoe UI", 10, "bold"), bg="#7f8c8d", fg="white").pack()
        
        self.lbl_risk = tk.Label(self.frm_risk, text="OFFLINE", 
                                 font=("Arial", 32, "bold"), bg="#7f8c8d", fg="white")
        self.lbl_risk.pack(pady=5)

    def create_metric_cards(self):
        # Helper to make cards
        def make_card(parent, title, unit):
            frm = tk.Frame(parent, bg="#34495e", pady=10, padx=10)
            frm.pack(fill=tk.X, pady=5, padx=10)
            
            tk.Label(frm, text=title, font=("Segoe UI", 10), bg="#34495e", fg="#bdc3c7", anchor="w").pack(fill=tk.X)
            lbl = tk.Label(frm, text="--", font=self.font_value, bg="#34495e", fg="white", anchor="e")
            lbl.pack(fill=tk.X)
            tk.Label(frm, text=unit, font=("Segoe UI", 8), bg="#34495e", fg="#95a5a6", anchor="e").pack(fill=tk.X)
            return lbl

        self.lbl_heads = make_card(self.data_panel, "CROWD DENSITY (HEADS)", "People detected")
        self.lbl_motion = make_card(self.data_panel, "PANIC VELOCITY", "Avg pixel movement/sec")
        self.lbl_entropy = make_card(self.data_panel, "TURBULENCE (ENTROPY)", "Chaos metric (0.0 - 3.0)")
        self.lbl_frame = make_card(self.data_panel, "FRAME ID", "Sequence number")

    def update_clock(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self.lbl_clock.config(text=now)
        
        # Check for timeout (Watchdog)
        if time.time() - self.last_packet_time > 3.0 and self.last_packet_time > 0:
             self.set_status("⚠️ No Data Stream (Check Jetson)", "orange")
             
        if self.running:
            self.root.after(1000, self.update_clock)

    def set_status(self, text, color="white"):
        self.status_bar.config(text=f" {text}", bg="#34495e", fg=color)

    def network_loop(self):
        while self.running:
            try:
                self.set_status(f"Connecting to {SERVER_IP}:{SERVER_PORT}...", "yellow")
                self.lbl_video.config(text="CONNECTING...", bg="black")
                
                client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client.settimeout(5)
                client.connect((SERVER_IP, SERVER_PORT))
                
                self.set_status(f"✅ Connected to FALCON Core", "#27ae60")
                self.last_packet_time = time.time()
                
                while self.running:
                    # 1. Read JSON Size
                    raw_size = self.recvall(client, 4)
                    if not raw_size: break
                    size = struct.unpack("!L", raw_size)[0]
                    
                    # 2. Read JSON
                    json_raw = self.recvall(client, size)
                    if not json_raw: break
                    data = json.loads(zlib.decompress(json_raw))
                    
                    # 3. Read Image Size
                    raw_size = self.recvall(client, 4)
                    if not raw_size: break
                    size = struct.unpack("!L", raw_size)[0]
                    
                    # 4. Read Image
                    img_raw = self.recvall(client, size)
                    if not img_raw: break
                    frame = cv2.imdecode(np.frombuffer(img_raw, np.uint8), 1)
                    
                    self.last_packet_time = time.time()
                    self.update_ui(data, frame)
                    
            except Exception as e:
                self.set_status(f"❌ Connection Failed: {e}", "#c0392b")
                time.sleep(2)
            finally:
                try: client.close()
                except: pass

    def recvall(self, sock, n):
        data = b''
        while len(data) < n:
            try:
                chunk = sock.recv(n - len(data))
                if not chunk: return None
                data += chunk
            except socket.timeout:
                return None
        return data

    def update_ui(self, data, frame):
        # Image Resize & Display
        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            
            # Responsive Resize
            w_canvas = self.video_frame.winfo_width()
            h_canvas = self.video_frame.winfo_height()
            
            if w_canvas > 10 and h_canvas > 10:
                # Keep aspect ratio
                img_w, img_h = img.size
                ratio = min(w_canvas/img_w, h_canvas/img_h)
                new_size = (int(img_w*ratio), int(img_h*ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            tk_img = ImageTk.PhotoImage(image=img)
            self.last_tk_img = tk_img # Keep ref
        else:
            tk_img = None

        # Risk Color Logic
        risk = data.get('risk', 'LOW')
        bg_color = "#27ae60" # Green
        status_text = "NORMAL"
        
        if risk == "MEDIUM": 
            bg_color = "#f39c12" # Orange
            status_text = "CAUTION"
        elif risk == "HIGH": 
            bg_color = "#c0392b" # Red
            status_text = "CRITICAL"

        # Thread-safe Update
        def _update():
            if tk_img:
                self.lbl_video.config(image=tk_img, text="")
                self.lbl_video.image = tk_img
            
            # Update Risk Panel
            self.lbl_risk.config(text=status_text, bg=bg_color)
            self.frm_risk.config(bg=bg_color)
            
            # Update Metrics
            self.lbl_heads.config(text=str(data.get('heads', 0)))
            self.lbl_motion.config(text=f"{data.get('motion', 0.0):.2f}")
            self.lbl_entropy.config(text=f"{data.get('entropy', 0.0):.2f}")
            self.lbl_frame.config(text=str(data.get('frame', 0)))
            
        self.root.after(0, _update)

if __name__ == "__main__":
    root = tk.Tk()
    app = FalconDashboard(root)
    root.mainloop()