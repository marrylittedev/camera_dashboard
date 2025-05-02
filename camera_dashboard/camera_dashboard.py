import cv2
import numpy as np
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import os

# Roboflow setup
rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("trained-q5iwo")
model = project.version(2).model

# HSV color thresholds
criollo_lower = np.array([0, 10, 180])
criollo_upper = np.array([15, 80, 255])
forastero_lower = np.array([130, 50, 50])
forastero_upper = np.array([170, 255, 255])
trinitario_lower = np.array([10, 50, 100])
trinitario_upper = np.array([30, 255, 255])
min_match_threshold = 10.0

# Detection counters
counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0, "Unknown": 0}
last_pred_time = 0

# Tkinter GUI
root = tk.Tk()
root.title("Cacao Detection")
root.geometry("900x600")
root.configure(bg="#2E2E2E")

# Create a frame for video and dashboard to be centered
frame = tk.Frame(root, bg="#2E2E2E")
frame.pack(expand=True)

# Left: Video frame (fixed size)
video_label = tk.Label(frame, bd=2, relief="solid")
video_label.grid(row=0, column=0, padx=10, pady=10)

# Right: Dashboard
dashboard = tk.Frame(frame, bg="#2E2E2E", width=300)
dashboard.grid(row=0, column=1, sticky="ns", padx=10)

# Dashboard variables
criollo_var = tk.StringVar(value="Criollo: 0")
forastero_var = tk.StringVar(value="Forastero: 0")
trinitario_var = tk.StringVar(value="Trinitario: 0")
unknown_var = tk.StringVar(value="Unknown: 0")
detected_type_var = tk.StringVar(value="Detected: Waiting")

# Dashboard content
tk.Label(dashboard, text="🧠 Detection Summary", font=("Arial", 16, "bold"),
         fg="white", bg="#2E2E2E").pack(pady=10)
tk.Label(dashboard, textvariable=criollo_var, font=("Arial", 12),
         fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=forastero_var, font=("Arial", 12),
         fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=trinitario_var, font=("Arial", 12),
         fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=unknown_var, font=("Arial", 12),
         fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=detected_type_var, font=("Arial", 14, "bold"),
         fg="#00BFFF", bg="#2E2E2E").pack(pady=(10, 0))

tk.Button(dashboard, text="❌ Exit", font=("Arial", 12), command=lambda: root.quit(),
          bg="#FF6347", fg="white", relief="flat", padx=15, pady=5).pack(pady=20)

# Show logo splash before camera loads
def show_logo():
    try:
        logo_image = Image.open("cacao.jpg")
        logo_image = logo_image.resize((640, 480), Image.Resampling.LANCZOS)
        logo_tk = ImageTk.PhotoImage(logo_image)
        video_label.configure(image=logo_tk)
        video_label.image = logo_tk
    except Exception as e:
        print(f"Logo load failed: {e}")

show_logo()
root.update()  # Force update to show logo immediately

# Camera init
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Wait until camera is ready (in background)
def wait_for_camera_start():
    for _ in range(30):  # Try for 3 seconds
        ret, _ = cap.read()
        if ret:
            root.after(100, update_frame)
            return
        time.sleep(0.1)
    print("Camera failed to start.")

# Frame prediction
def predict_and_update(frame):
    global last_pred_time
    ts = time.time()
    if ts - last_pred_time < 1.5:
        return
    last_pred_time = ts

    cv2.imwrite("../frame.jpg", frame)
    try:
        predictions = model.predict("frame.jpg", confidence=40, overlap=30).json()
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    for k in counts:
        counts[k] = 0

    for pred in predictions.get("predictions", []):
        x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
        x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
        x2, y2 = min(x + w // 2, frame.shape[1]), min(y + h // 2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        masks = {
            "Criollo": cv2.inRange(hsv, criollo_lower, criollo_upper),
            "Forastero": cv2.inRange(hsv, forastero_lower, forastero_upper),
            "Trinitario": cv2.inRange(hsv, trinitario_lower, trinitario_upper),
        }

        color_label = "Unknown"
        for name, mask in masks.items():
            ratio = (cv2.countNonZero(mask) / (crop.size / 3)) * 100
            if ratio > min_match_threshold:
                color_label = name
                counts[name] += 1
                break
        else:
            counts["Unknown"] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{pred['class']} | {color_label}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    criollo_var.set(f"Criollo: {counts['Criollo']}")
    forastero_var.set(f"Forastero: {counts['Forastero']}")
    trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
    unknown_var.set(f"Unknown: {counts['Unknown']}")
    detected_type_var.set(f"Detected: {max(counts, key=counts.get)}")

# Frame loop
def update_frame():
    ret, frame = cap.read()
    if ret:
        threading.Thread(target=predict_and_update, args=(frame.copy(),), daemon=True).start()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        # Set fixed size for the video feed to prevent movement
        img = img.resize((640, 480), Image.Resampling.LANCZOS)

        # Convert back to PhotoImage for Tkinter
        resized_img_tk = ImageTk.PhotoImage(img)

        video_label.configure(image=resized_img_tk)
        video_label.image = resized_img_tk
    root.after(30, update_frame)

# Start camera check in background
threading.Thread(target=wait_for_camera_start, daemon=True).start()

# Main loop
root.mainloop()
cap.release()
cv2.destroyAllWindows()
