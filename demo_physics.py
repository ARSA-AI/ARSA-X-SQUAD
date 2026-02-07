#!/usr/bin/env python3
"""
Trillion Physics Demo v13 - "The Comet Tail"
ARSA AI

Usage:
  python3 demo_physics.py --video website/assets/test1.mp4 --webcam 0
"""

import cv2
import numpy as np
import time
import argparse
import sys
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# Global frames for streaming
outputFrame = None
debugFrame = None
webcamFrame = None
lock = threading.Lock()

class PhysicsEngine:
    def __init__(self, mode="video"):
        self.mode = mode # 'video' or 'webcam'
        self.prev_gray = None
        self.features = []
        # Tweak params for better performance
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.foe_smooth = None 
        self.alpha = 0.1 # Faster convergence for smoothness
        self.foe_history = []
        self.max_history = 40 # Short, responsive "Comet Tail" for vanishing effect
        
    def calculate_ttc(self, flow_vectors, center):
        if not flow_vectors: return float('inf')
        divergences = []
        cx, cy = center
        for (x, y, u, v) in flow_vectors:
            rx, ry = x - cx, y - cy
            r = np.sqrt(rx**2 + ry**2)
            if r < 20: continue 
            v_radial = (u * rx + v * ry) / r
            if v_radial > 0.5: 
                ttc = r / v_radial
                if ttc < 10.0: divergences.append(ttc)
        if not divergences: return float('inf')
        return np.median(divergences) / 30.0 
        
    def process(self, frame):
        # Resize based on mode
        target_size = (800, 600) if self.mode == "video" else (640, 480)
        frame = cv2.resize(frame, target_size) 
        h, w = frame.shape[:2]
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        ar_view = frame.copy()
        
        # Only create physics view if calculating physics (Main Video)
        physics_view = None
        if self.mode == "video":
            physics_view = np.zeros((h, w, 3), dtype=np.uint8)
            # Grid
            step = 50
            for x in range(0, w, step): cv2.line(physics_view, (x, 0), (x, h), (40, 40, 40), 1)
            for y in range(0, h, step): cv2.line(physics_view, (0, y), (w, y), (40, 40, 40), 1)
        
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if p0 is not None: self.features = p0
            return ar_view, physics_view

        # Optical Flow
        flow_vecs = []
        avg_expansion = 0
        if len(self.features) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.features, None, **self.lk_params)
            good_new = p1[st==1]
            good_old = self.features[st==1]
            
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                u, v = a-c, b-d
                dist = np.sqrt(u**2 + v**2)
                if dist > 0.5:
                    flow_vecs.append((a, b, u, v))
                    avg_expansion += (u + v)
                    
                    # Draw Vectors
                    if self.mode == "video":
                        cv2.arrowedLine(ar_view, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1, tipLength=0.3)
                        cv2.arrowedLine(physics_view, (int(c), int(d)), (int(c+u*3), int(d+v*3)), (0, 255, 255), 1)
                    else:
                        cv2.arrowedLine(ar_view, (int(c), int(d)), (int(a), int(b)), (0, 255, 255), 1, tipLength=0.3)

            # FOE Calculation (DISABLED FOR WEBCAM as requested)
            if self.mode == "video":
                count = len(flow_vecs)
                
                if count > 5:
                    avg_u = np.mean([f[2] for f in flow_vecs])
                    avg_v = np.mean([f[3] for f in flow_vecs])
                    
                    cx, cy = w//2, h//2
                    target_x = cx - avg_u * 40
                    target_y = cy - avg_v * 40
                    target_x = np.clip(target_x, 0, w)
                    target_y = np.clip(target_y, 0, h)
                    
                    if self.foe_smooth is None: self.foe_smooth = np.array([target_x, target_y], dtype=np.float32)
                    self.foe_smooth = (1 - self.alpha) * self.foe_smooth + self.alpha * np.array([target_x, target_y], dtype=np.float32)
                    
                # Draw FOE & TTC
                if self.foe_smooth is not None:
                    foe_x, foe_y = int(self.foe_smooth[0]), int(self.foe_smooth[1])
                    
                    # Safety Tunnel
                    pts = np.array([[w//2 - 250, h], [w//2 + 250, h], [foe_x + 60, foe_y], [foe_x - 60, foe_y]], np.int32)
                    cv2.polylines(ar_view, [pts], True, (0, 255, 255), 1)
                    cv2.polylines(physics_view, [pts], True, (0, 255, 0), 2)
                    
                    # History Trail (COMET TAIL EFFECT)
                    self.foe_history.append((foe_x, foe_y))
                    if len(self.foe_history) > self.max_history: self.foe_history.pop(0)
                    
                    # Gradient Drawing
                    for k in range(1, len(self.foe_history)):
                        # Opacity or "Vanishing" Simulation via Thickness/Color intensity
                        # Normalize progress 0.0 -> 1.0
                        progress = k / len(self.foe_history)
                        thickness = max(1, int(4 * progress))
                        # Color: Dark Orange -> Bright Yellow
                        color_val = int(255 * progress)
                        color = (0, color_val//2, color_val) # BGR
                        
                        cv2.line(physics_view, self.foe_history[k-1], self.foe_history[k], color, thickness)

                    
                    # Crosshair
                    cv2.line(ar_view, (foe_x-20, foe_y), (foe_x+20, foe_y), (0, 0, 255), 2)
                    cv2.line(ar_view, (foe_x, foe_y-20), (foe_x, foe_y+20), (0, 0, 255), 2)

                    # TTC
                    ttc = self.calculate_ttc(flow_vecs, (foe_x, foe_y))
                    
                    # HUD Rendering
                    hud_x = 20
                    hud_y = 50
                    line_h = 25
                    font = cv2.FONT_HERSHEY_PLAIN
                    
                    # Logic Box
                    cv2.rectangle(physics_view, (10, 10), (350, 180), (30, 30, 30), -1)
                    cv2.rectangle(physics_view, (10, 10), (350, 180), (100, 100, 100), 1)
                    
                    cv2.putText(physics_view, "PHYSICS CORE V13", (hud_x, hud_y - 15), font, 1.2, (0, 255, 255), 1)
                    cv2.putText(physics_view, f"Flow Vectors: {count}", (hud_x, hud_y + line_h*1), font, 1.0, (200, 200, 200), 1)
                    cv2.putText(physics_view, f"Expansion Vel: {avg_expansion/max(1,count):.2f}", (hud_x, hud_y + line_h*2), font, 1.0, (200, 200, 200), 1)
                    cv2.putText(physics_view, "TTC = r / v_radial", (hud_x, hud_y + line_h*3), font, 1.0, (150, 150, 255), 1)
                    
                    # TTC Color
                    t_clr = (0, 255, 0)
                    if ttc < 2.5: t_clr = (0, 0, 255)
                    elif ttc < 5.0: t_clr = (0, 255, 255)
                    
                    cv2.putText(physics_view, f"Result: {ttc:.2f}s", (hud_x, hud_y + line_h*4), font, 1.5, t_clr, 2)
                    
                    decision = "SAFE"
                    if ttc < 2.5: decision = "CRITICAL (BRAKE)"
                    elif ttc < 5.0: decision = "WARNING (SLOW)"
                    
                    cv2.putText(physics_view, f"Logic: {decision}", (hud_x, hud_y + line_h*5), font, 1.2, t_clr, 1)
                    cv2.putText(ar_view, f"TTC: {ttc:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, t_clr, 3)

            self.features = good_new.reshape(-1, 1, 2)

        # Maintenance
        if len(self.features) < 150:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if p0 is not None:
                self.features = np.concatenate((self.features, p0), axis=0) if len(self.features) > 0 else p0
        
        self.prev_gray = frame_gray
        return ar_view, physics_view

# ==========================================================
# STREAMING
# ==========================================================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer): pass

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        target_frame = None
        
        if self.path == '/stream.mjpg': target_frame = 'output'
        elif self.path == '/debug.mjpg': target_frame = 'debug'
        elif self.path == '/webcam.mjpg': target_frame = 'webcam'
        else:
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        
        try:
            while True:
                with lock:
                    if target_frame == 'output': frm = outputFrame
                    elif target_frame == 'debug': frm = debugFrame
                    elif target_frame == 'webcam': frm = webcamFrame
                    
                    if frm is None: 
                        time.sleep(0.005)
                        continue
                        
                    (flag, encodedImage) = cv2.imencode(".jpg", frm, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                    if not flag: continue
                
                self.wfile.write(b"--jpgboundary\r\n")
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(encodedImage)))
                self.end_headers()
                self.wfile.write(bytearray(encodedImage))
                self.wfile.write(b"\r\n")
        except: return

def start_server():
    server = ThreadedHTTPServer(('0.0.0.0', 5000), StreamHandler)
    server.serve_forever()

# ==========================================================
# MAIN APP
# ==========================================================
class DemoApp:
    def __init__(self, video_path, webcam_id):
        self.cap_vid = cv2.VideoCapture(video_path)
        self.cap_web = cv2.VideoCapture(webcam_id)
        
        self.eng_vid = PhysicsEngine(mode="video")
        self.eng_web = PhysicsEngine(mode="webcam")
        
        if not self.cap_vid.isOpened(): print(f"❌ Failed to open video: {video_path}")
        if not self.cap_web.isOpened(): print(f"❌ Failed to open webcam: {webcam_id}")

    def run(self):
        global outputFrame, debugFrame, webcamFrame
        
        def read_loop_video():
            global outputFrame, debugFrame
            while True:
                ret, frame = self.cap_vid.read()
                if not ret:
                    self.cap_vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                ar, phys = self.eng_vid.process(frame)
                
                with lock:
                    outputFrame = ar
                    debugFrame = phys
                
                time.sleep(0.005)

        def read_loop_webcam():
            global webcamFrame
            while True:
                ret, frame = self.cap_web.read()
                if not ret: continue
                
                ar, _ = self.eng_web.process(frame)
                
                with lock:
                    webcamFrame = ar
                    
                time.sleep(0.01)

        t_vid = threading.Thread(target=read_loop_video)
        t_vid.daemon = True
        t_vid.start()
        
        t_web = threading.Thread(target=read_loop_webcam)
        t_web.daemon = True
        t_web.start()
        
        print("📡 Server Started. Optimization: MAX PERFORMANCE.")
        start_server()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='website/assets/test1.mp4')
    parser.add_argument('--webcam', default=0)
    parser.add_argument('--video2', default='')
    args = parser.parse_args()
    
    try: wc = int(args.webcam)
    except: wc = args.webcam
    
    app = DemoApp(args.video, wc)
    app.run()
