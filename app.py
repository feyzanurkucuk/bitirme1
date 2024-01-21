import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import numpy as np
import pathlib


pathlib.PosixPath = pathlib.WindowsPath
# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#model = torch.hub.load('ultralytics/yolov5','custom',path= '../bitirme/yolov5/runs/train/exp4/weights/best.pt', force_reload=True)

model = torch.hub.load('ultralytics/yolov5','custom',path= 'weights_last/best.pt', force_reload=True)
cap = cv2.VideoCapture(0)
running = False

# Create a Tkinter window
root = tk.Tk()
root.title("Nail Biting Detection App")

# Create a Label to display the video feed
lblVideo = tk.Label(root)
lblVideo.grid(row=0, column=0, padx=10, pady=10)

# Create a ProgressBar for prediction percentage
progress_var = tk.DoubleVar()
prediction_progress = ttk.Progressbar(root, variable=progress_var, length=200, mode='determinate')
prediction_progress.grid(row=0, column=1, padx=10, pady=10)

# Create a label for additional details
details_label = tk.Label(root, text="Additional Details")
details_label.grid(row=1, column=0, columnspan=2, pady=10)

def start_prediction():
    global running
    if button_text.get() == "Start":
        print("Starting prediction...")
        running = True
        update_gui()
    else:
        print("Stopping prediction...")
        running = False

    button_text.set("Stop" if button_text.get() == "Start" else "Start")

button_text = tk.StringVar()
button_text.set("Start")

start_button = tk.Button(root, textvariable=button_text, command=start_prediction)
start_button.grid(row=2, column=0, columnspan=2, pady=10)

def update_gui():
    if running:
        ret, frame = cap.read()
        results = model(frame)

        labels = results.names
        confidences = results.xyxy[0][:, 4]
        num_detections = len(results.xyxy[0])
        confidence_threshold = 0.5

        for i in range(num_detections):
            confidence = confidences[i]
            
            if confidence > confidence_threshold:
                label_id = int(results.xyxy[0][i][5])
                label = labels[label_id]
                if label == 'nail-biting':
                    print(f'Detected label: {label}, Confidence: {confidence:.2f}')
        
        # Convert the OpenCV image to a format compatible with Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)

        # Update the Label with the new image
        lblVideo.configure(image=img)
        lblVideo.image = img

        cv2.imshow('YOLO', np.squeeze(results.render()))

        if button_text.get() == "Start":
            root.after(10, update_gui)
        if cv2.waitKey(10) & 0XFF == ord('q'):
            stop_prediction()

def stop_prediction():
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()
    print("Prediction stopped.")

# Start the Tkinter main loop
root.mainloop()