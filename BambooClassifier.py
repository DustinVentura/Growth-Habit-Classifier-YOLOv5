import sys
from pathlib import Path
import uuid
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
import numpy as np

# ====================== ADD YOLOV5 TO PATH ======================
YOLOv5_DIR = Path(__file__).parent / 'yolov5'
sys.path.append(str(YOLOv5_DIR))

# ====================== SAFE MODEL LOADING ======================
from yolov5.models.yolo import DetectionModel
torch.serialization.add_safe_globals([DetectionModel])

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.augmentations import letterbox

# ====================== LOAD MODEL ======================

# Change to your File Directory where the model is located
model_path = r'...yolov5\runs\train\exp2\weights\best.pt'

try:
    model = DetectMultiBackend(weights=model_path, device='cpu')
    print("✅ Model Loaded Successfully using DetectMultiBackend")
except Exception as e:
    print(f"DetectMultiBackend failed: {e}. Using fallback...")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = ckpt['model'].float().fuse().eval() if isinstance(ckpt, dict) else ckpt
    print("✅ Model loaded using fallback method")

# ====================== GLOBAL VARIABLES ======================
processed_image_path = 'detected/result.jpg'
selected_image_path = None

# ====================== AUTO CREATE 'detected' FOLDER ======================
Path('detected').mkdir(parents=True, exist_ok=True)

# ====================== GUI SETUP ======================
root = tk.Tk()
root.title("Bamboo Growth Habit Classification")
root.geometry("700x750")
root.configure(bg="#f0f0f0")

main_frame = tk.Frame(root, bg="#f0f0f0")
main_frame.pack(expand=True, fill="both", padx=20, pady=20)

classification_result = tk.StringVar()

# ====================== HELPER FUNCTIONS ======================
def return_to_start():
    for widget in main_frame.winfo_children():
        widget.destroy()
    show_start_screen()


def show_start_screen():
    for widget in main_frame.winfo_children():
        widget.destroy()

    title = tk.Label(main_frame, text="Bamboo Growth Habit Classification", 
                     font=("Helvetica", 22, "bold"), bg="#f0f0f0")
    title.pack(pady=50)

    subtitle = tk.Label(main_frame, text="Choose a bamboo image to classify growth habit", 
                        font=("Helvetica", 12), bg="#f0f0f0", fg="#555")
    subtitle.pack(pady=10)

    choose_btn = tk.Button(main_frame, text="Choose Photo from Computer", 
                           command=choose_image,
                           font=("Helvetica", 14), bg="#4CAF50", fg="white", 
                           padx=40, pady=15, relief="flat")
    choose_btn.pack(pady=30)


def choose_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if selected_image_path:
        display_selected_image(selected_image_path)


def display_selected_image(image_path):
    for widget in main_frame.winfo_children():
        widget.destroy()

    img = Image.open(image_path)
    img = img.resize((500, 500), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    label = tk.Label(main_frame, image=img_tk, bg="#f0f0f0")
    label.image = img_tk
    label.pack(pady=20)

    tk.Button(main_frame, text="Use This Photo", 
              command=lambda: use_this_photo(image_path),
              font=("Helvetica", 13, "bold"), bg="#4CAF50", fg="white", 
              padx=35, pady=12).pack(pady=15)

    tk.Button(main_frame, text="Choose Another Photo", 
              command=choose_image,
              font=("Helvetica", 12), bg="#2196F3", fg="white", 
              padx=30, pady=10).pack(pady=8)

    tk.Button(main_frame, text="Return To Start", 
              command=return_to_start,
              font=("Helvetica", 12), bg="#f44336", fg="white", 
              padx=30, pady=10).pack(pady=8)


# ====================== MAIN PROCESSING ======================
def use_this_photo(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            classification_result.set("Error: Could not read the image.")
            return

        img_resized = letterbox(image, (640, 640), stride=32, auto=True)[0]
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        results = model(img_tensor)
        predictions = non_max_suppression(results, conf_thres=0.25, iou_thres=0.45)[0]

        detected_classes = []
        orig_height, orig_width = image.shape[:2]

        for det in predictions:
            if len(det) == 0:
                continue

            conf = float(det[4]) * 100

            x1, y1, x2, y2 = det[:4].tolist()
            scale_x = orig_width / 640.0
            scale_y = orig_height / 640.0

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            class_name = model.names[int(det[5])]
            label = f"{class_name} ({conf:.2f}%)"
            detected_classes.append(label)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(image, label, (x1, y1 - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Save processed image
        processed_image_path = f'detected/result_{uuid.uuid4().hex[:8]}.jpg'
        cv2.imwrite(processed_image_path, image)

        if detected_classes:
            result_text = f"Detected: {', '.join(set(detected_classes))}"
        else:
            result_text = "No detections found"

        classification_result.set(result_text)
        display_selected_image(processed_image_path)

        tk.Label(main_frame, textvariable=classification_result, 
                font=("Helvetica", 14, "bold"), pady=20, bg="#f0f0f0", fg="#1a3c34").pack()

        tk.Button(main_frame, text="Save Result", command=save_result,
                  font=("Helvetica", 12), bg="#2196F3", fg="white", padx=35, pady=10).pack(pady=10)
        
        tk.Button(main_frame, text="Return To Start", command=return_to_start,
                  font=("Helvetica", 12), bg="#f44336", fg="white", padx=35, pady=10).pack(pady=10)

    except Exception as e:
        print(f"Error in use_this_photo: {e}")
        classification_result.set(f"Error: {str(e)}")


def save_result():
    try:
        save_path = filedialog.asksaveasfilename(
            defaultextension=".jpg", 
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if save_path:
            img = cv2.imread(processed_image_path)
            cv2.imwrite(save_path, img)
            print(f"Result saved at {save_path}")
    except Exception as e:
        print(f"Save error: {e}")


# ====================== START THE APP ======================
show_start_screen()
root.mainloop()