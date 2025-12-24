import serial
import json
import struct
import tkinter as tk
from tkinter import ttk, messagebox, font, filedialog
from PIL import Image, ImageTk, ImageGrab
import threading
import time
import io
import winsound
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# --- CONFIGURATION ---
COM_PORT = 'COM3'  # <--- CHANGE THIS to your Laptop's Radio Port
BAUD_RATE = 57600

# COMMANDS
CMD_PING = 0x10
CMD_START = 0x11
CMD_STOP = 0x12
CMD_SHUTDOWN = 0x14
CMD_REQ_IMG = 0x05

# THEME
C_BG_MAIN = "#1e272e"
C_BG_PANEL = "#2d3436"
C_ACCENT = "#0fb9b1"
C_WARN = "#fa8231"
C_DANGER = "#ff3f34"
C_TEXT_MAIN = "#ffffff"   
C_TEXT_SUB = "#b2bec3"

class RadioDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title(f"FALCON COMMAND CENTER - {COM_PORT}")
        self.root.geometry("1400x850")
        self.root.configure(bg=C_BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # State Variables
        self.stats_labels = {}
        self.ser = None
        self.running = True
        self.image_buffer = {}
        self.lock = threading.Lock()
        
        # History for Graph
        self.history_len = 60
        self.history_heads = deque([0]*self.history_len, maxlen=self.history_len)

        # Custom Fonts
        self.f_head = font.Font(family="Segoe UI", size=20, weight="bold")
        self.f_sub = font.Font(family="Segoe UI", size=10)
        self.f_val = font.Font(family="Consolas", size=22, weight="bold")
        self.f_btn = font.Font(family="Segoe UI", size=10, weight="bold")

        self.build_ui()
        threading.Thread(target=self.radio_loop, daemon=True).start()

    def on_close(self):
        self.running = False
        if self.ser:
            try: self.ser.close()
            except: pass
        self.root.destroy()

    def build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=C_BG_PANEL, height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)
        tk.Label(header, text="🦅 FALCON AI COMMAND", font=self.f_head, bg=C_BG_PANEL, fg=C_ACCENT).pack(side=tk.LEFT, padx=20)
        self.lbl_status = tk.Label(header, text="DISCONNECTED", font=("Segoe UI", 12, "bold"), bg=C_BG_PANEL, fg=C_DANGER)
        self.lbl_status.pack(side=tk.RIGHT, padx=20)

        # Main Layout
        container = tk.Frame(self.root, bg=C_BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Column 1: Stats & Graph
        col1 = tk.Frame(container, bg=C_BG_MAIN, width=320)
        col1.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        self.frm_metrics = tk.Frame(col1, bg=C_BG_MAIN)
        self.frm_metrics.pack(fill=tk.X)
        self.create_card("CROWD COUNT", "0", "h")
        self.create_card("RISK LEVEL", "L", "r_text") # Placeholder
        
        tk.Label(col1, text="📉 DENSITY TREND", font=("Segoe UI", 10, "bold"), bg=C_BG_MAIN, fg=C_TEXT_SUB).pack(pady=(20, 5), anchor="w")
        self.fig, self.ax = plt.subplots(figsize=(3.5, 2.5), dpi=100)
        self.fig.patch.set_facecolor(C_BG_PANEL)
        self.ax.set_facecolor(C_BG_PANEL)
        self.line_heads, = self.ax.plot([], [], color=C_ACCENT, lw=2)
        self.ax.axis('off') # Cleaner look
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=col1)
        self.canvas_graph.get_tk_widget().pack(fill=tk.X)

        # Column 2: Controls & Map
        col2 = tk.Frame(container, bg=C_BG_MAIN)
        col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)
        
        ctrl_panel = tk.Frame(col2, bg=C_BG_PANEL, pady=10)
        ctrl_panel.pack(fill=tk.X, pady=(0, 15))
        btn_frm = tk.Frame(ctrl_panel, bg=C_BG_PANEL)
        btn_frm.pack(fill=tk.X, padx=10)
        self.create_sys_btn(btn_frm, "▶ START", CMD_START, "#27ae60")
        self.create_sys_btn(btn_frm, "⏸ STOP", CMD_STOP, "#f39c12")
        self.create_sys_btn(btn_frm, "💀 SHUTDOWN", CMD_SHUTDOWN, "#c0392b")

        self.frm_map = tk.Frame(col2, bg="black", bd=2, relief="sunken")
        self.frm_map.pack(fill=tk.BOTH, expand=True)
        self.lbl_map = tk.Label(self.frm_map, text="MAP VIEW (Click Import)", bg="black", fg=C_TEXT_SUB)
        self.lbl_map.pack(fill=tk.BOTH, expand=True)
        tk.Button(col2, text="🗺️ IMPORT MAP", command=self.load_map).pack(fill=tk.X)

        # Column 3: Visuals
        col3 = tk.Frame(container, bg=C_BG_MAIN, width=320)
        col3.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        self.frm_risk = tk.Frame(col3, bg=C_BG_PANEL, pady=15)
        self.frm_risk.pack(fill=tk.X, pady=(0, 15))
        tk.Label(self.frm_risk, text="THREAT LEVEL", font=("Segoe UI", 10, "bold"), bg=C_BG_PANEL, fg=C_TEXT_SUB).pack()
        self.lbl_risk = tk.Label(self.frm_risk, text="STANDBY", font=("Segoe UI", 24, "bold"), bg=C_BG_PANEL, fg=C_TEXT_SUB)
        self.lbl_risk.pack()

        self.lbl_img = tk.Label(col3, text="NO FEED", bg="black", fg="#636e72", height=15)
        self.lbl_img.pack(fill=tk.X)

    def create_card(self, title, default, key):
        card = tk.Frame(self.frm_metrics, bg=C_BG_PANEL, pady=5, padx=10)
        card.pack(pady=3, fill=tk.X)
        tk.Label(card, text=title, font=self.f_sub, bg=C_BG_PANEL, fg=C_ACCENT).pack(anchor="w")
        lbl = tk.Label(card, text=default, font=self.f_val, bg=C_BG_PANEL, fg=C_TEXT_MAIN)
        lbl.pack(anchor="e")
        self.stats_labels[key] = lbl

    def create_sys_btn(self, parent, text, cmd, color):
        tk.Button(parent, text=text, bg=color, fg="white", font=self.f_btn,
                  command=lambda: self.send_command(cmd)).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    def send_command(self, cmd_byte):
        if self.ser and self.ser.is_open:
            try:
                with self.lock: self.ser.write(struct.pack('B', cmd_byte))
            except: pass

    def load_map(self):
        path = filedialog.askopenfilename()
        if path:
            img = Image.open(path)
            img.thumbnail((600, 400))
            tk_img = ImageTk.PhotoImage(img)
            self.lbl_map.config(image=tk_img, text="")
            self.lbl_map.image = tk_img

    def radio_loop(self):
        while self.running:
            try:
                if not self.ser:
                    self.ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
                    self.lbl_status.config(text="CONNECTED", fg="#00b894")
                
                # Header Check
                if self.ser.read(1) != b'\xfa': continue
                if self.ser.read(1) != b'\xce': continue

                packet_type = struct.unpack('B', self.ser.read(1))[0]
                length = struct.unpack('!H', self.ser.read(2))[0]
                payload = self.ser.read(length)
                self.ser.read(1) # Checksum

                if packet_type == 0x01: self.handle_json(payload)
                elif packet_type == 0x02: self.handle_chunk(payload)
                elif packet_type == 0x03: self.handle_status(payload)
            except:
                if self.ser: 
                    try: self.ser.close()
                    except: pass
                self.ser = None
                self.lbl_status.config(text="CONNECTING...", fg=C_WARN)
                time.sleep(2)

    def handle_json(self, payload):
        try:
            data = json.loads(payload.decode('utf-8'))
            self.history_heads.append(data.get('h', 0))
            def _update():
                if 'h' in self.stats_labels: self.stats_labels['h'].config(text=str(data.get('h', 0)))
                r = data.get('r', 'L')
                colors = {'L': ('LOW', C_ACCENT), 'M': ('CAUTION', C_WARN), 'H': ('CRITICAL', C_DANGER)}
                txt, col = colors.get(r, ('UNK', 'gray'))
                self.lbl_risk.config(text=txt, fg=col)
                if r == 'H': winsound.Beep(1000, 200)
                
                # Update Graph
                self.line_heads.set_data(range(len(self.history_heads)), list(self.history_heads))
                self.ax.set_ylim(0, max(max(self.history_heads, default=10), 10) * 1.2)
                self.ax.set_xlim(0, self.history_len)
                self.canvas_graph.draw()
            self.root.after(0, _update)
        except: pass

    def handle_chunk(self, payload):
        if len(payload) < 3: return
        chunk_id, total = payload[0], payload[1]
        self.image_buffer[chunk_id] = payload[2:]
        if len(self.image_buffer) == total:
            data = b''.join([self.image_buffer[i] for i in range(total)])
            try:
                img = Image.open(io.BytesIO(data))
                img.thumbnail((320, 240))
                tk_img = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: self.lbl_img.config(image=tk_img, text=""))
                self.root.after(0, lambda: setattr(self.lbl_img, 'image', tk_img))
            except: pass
            self.image_buffer = {}

    def handle_status(self, payload):
        try:
            msg = json.loads(payload.decode('utf-8')).get('s', '')
            self.root.after(0, lambda: messagebox.showinfo("System", msg))
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = RadioDashboard(root)
    root.mainloop()