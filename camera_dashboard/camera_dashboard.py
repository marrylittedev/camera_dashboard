import cv2
import numpy as np
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Roboflow setup
rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("trained-q5iwo")
model = project.version(2).model

# HSV color thresholds (you can fine-tune these)
criollo_lower = np.array([0, 10, 180])
criollo_upper = np.array([15, 80, 255])
forastero_lower = np.array([130, 50, 50])
forastero_upper = np.array([170, 255, 255])
trinitario_lower = np.array([5, 50, 100])  # Adjusted Trinitario lower bound
trinitario_upper = np.array([35, 255, 255])  # Adjusted Trinitario upper bound
min_match_threshold = 10.0

# Detection counters and state
counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0, "Unknown": 0}
last_pred_time = 0
last_predicted_frame = None
camera_ready = False

# Tkinter GUI setup
root = tk.Tk()
root.title("Cacao Detection")
root.geometry("900x600")
root.configure(bg="#2E2E2E")

frame = tk.Frame(root, bg="#2E2E2E")
frame.pack(expand=True)

video_label = tk.Label(frame, bd=2, relief="solid")
video_label.grid(row=0, column=0, padx=10, pady=10)

dashboard = tk.Frame(frame, bg="#2E2E2E", width=300)
dashboard.grid(row=0, column=1, sticky="ns", padx=10)

criollo_var = tk.StringVar(value="Criollo: 0")
forastero_var = tk.StringVar(value="Forastero: 0")
trinitario_var = tk.StringVar(value="Trinitario: 0")
unknown_var = tk.StringVar(value="Unknown: 0")
detected_type_var = tk.StringVar(value="Detected: Waiting")

tk.Label(dashboard, text="🧠 Detection Summary", font=("Arial", 16, "bold"), fg="white", bg="#2E2E2E").pack(pady=10)
tk.Label(dashboard, textvariable=criollo_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=forastero_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=trinitario_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=unknown_var, font=("Arial", 12), fg="white", bg="#2E2E2E").pack(pady=5)
tk.Label(dashboard, textvariable=detected_type_var, font=("Arial", 14, "bold"), fg="#00BFFF", bg="#2E2E2E").pack(pady=(10, 0))

tk.Button(dashboard, text="❌ Exit", font=("Arial", 12), command=lambda: root.quit(),
          bg="#FF6347", fg="white", relief="flat", padx=15, pady=5).pack(pady=20)

# Show logo while waiting for camera
def show_logo():
    try:
        logo_image = Image.open("cacao.jpg").resize((640, 480), Image.Resampling.LANCZOS)
        logo_tk = ImageTk.PhotoImage(logo_image)
        video_label.configure(image=logo_tk)
        video_label.image = logo_tk
    except Exception as e:
        print(f"Logo load failed: {e}")

# Prediction and update function with only one detection per frame
def predict_and_update(frame):
    global last_pred_time, last_predicted_frame
    last_pred_time = time.time()

    tmp_path = "frame.jpg"
    cv2.imwrite(tmp_path, frame)
    try:
        predictions = model.predict(tmp_path, confidence=20, overlap=30).json()  # Lower confidence threshold for faster detection
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    # Reset counts
    for k in counts:
        counts[k] = 0

    # Process only the first valid prediction
    first_detection_done = False  # Flag to track if we've processed the first detection

    for pred in predictions.get("predictions", []):
        if first_detection_done:  # If the first detection is already done, stop further processing
            break

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

        # New logic: pick the most dominant match
        best_match = "Unknown"
        best_ratio = 0

        for name, mask in masks.items():
            ratio = (cv2.countNonZero(mask) / (crop.size / 3)) * 100
            if ratio > best_ratio and ratio > min_match_threshold:
                best_ratio = ratio
                best_match = name

        counts[best_match] += 1
        color_label = best_match

        label_text = f"{pred['class']} | {color_label}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        first_detection_done = True  # Mark that we've processed one detection

    criollo_var.set(f"Criollo: {counts['Criollo']}")
    forastero_var.set(f"Forastero: {counts['Forastero']}")
    trinitario_var.set(f"Trinitario: {counts['Trinitario']}")
    unknown_var.set(f"Unknown: {counts['Unknown']}")
    detected_type_var.set(f"Detected: {max(counts, key=counts.get)}")

    last_predicted_frame = frame

# Frame update function with frame skipping
def update_frame():
    global last_pred_time, last_predicted_frame, camera_ready
    ret, frame = cap.read()
    if ret:
        if not camera_ready:
            camera_ready = True
            print("Camera ready.")

        # Skip frames to reduce load: process every 5th frame
        if time.time() - last_pred_time >= 3.0:  # Process every 3 seconds
            threading.Thread(target=predict_and_update, args=(frame.copy(),), daemon=True).start()

        display = last_predicted_frame if last_predicted_frame is not None else frame
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((640, 480), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        video_label.configure(image=img_tk)
        video_label.image = img_tk

    else:
        if not camera_ready:
            show_logo()

    root.after(50, update_frame)  # ~20 FPS (you can reduce it to 15 FPS or lower)

# Start camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Show logo until camera is ready
show_logo()
root.update()

# Start UI loop
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
