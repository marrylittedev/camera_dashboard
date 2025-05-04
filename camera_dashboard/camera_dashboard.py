import cv2
import numpy as np
from roboflow import Roboflow
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# Roboflow setup
rf = Roboflow(api_key="f4UBb9Y1BqAaVoiasTC1")
project = rf.workspace("cacaotrain").project("cacao_final")
version = project.version(1)
model = version.model  # Load model from Roboflow project

# Detection counters and state
counts = {"Criollo": 0, "Forastero": 0, "Trinitario": 0, "Unknown": 0}
last_pred_time = 0
last_predicted_frame = None
camera_ready = False
frame_skip = 3  # Process every 3rd frame (adjust as needed)
prediction_interval = 0.5  # Seconds between predictions

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

    # Resize frame to improve processing speed without sacrificing quality
    resized_frame = cv2.resize(frame, (640, 480))

    try:
        predictions = model.predict(resized_frame, confidence=20, overlap=30).json()  # Lower confidence threshold for faster detection
    except Exception as e:
        print(f"Prediction error: {e}")
        return

    # Reset counts
    for k in counts:
        counts[k] = 0

    # Process predictions
    for pred in predictions.get("predictions", []):
        x, y, w, h = map(int, [pred['x'], pred['y'], pred['width'], pred['height']])
        x1, y1 = max(x - w // 2, 0), max(y - h // 2, 0)
        x2, y2 = min(x + w // 2, frame.shape[1]), min(y + h // 2, frame.shape[0])
        crop = frame[y1:y2, x1:x2]

        # Directly use the predicted class label from YOLOv5
        label = pred['class']  # This is the label provided by YOLOv5, e.g., Criollo, Forastero, etc.
        counts[label] += 1  # Increment the count for the detected type

        label_text = f"{label}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

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

        # Skip frames to reduce load: process every 'frame_skip' frame
        if time.time() - last_pred_time >= prediction_interval:  # Adjust interval to process predictions at the specified interval
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
