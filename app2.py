from tkinter import Button, Label, messagebox
import tkinter
import cv2
from tkinter import Tk
from PIL import Image, ImageTk
import torch
import numpy as np
import pathlib


pathlib.PosixPath = pathlib.WindowsPath
# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = torch.hub.load('ultralytics/yolov5','custom',path= 'weights_last/best.pt', force_reload=True)

def inicial():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualize()


def visualize():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
   
            # Get YOLOv5 model predictions
            results = model(frame)

            # Process YOLOv5 results
            labels = results.names
            confidences = results.xyxy[0][:, 4]
            num_detections = len(results.xyxy[0])
            confidence_threshold = 0.6

            for i in range(num_detections):
                confidence = confidences[i]
                if confidence > confidence_threshold:
                    label_id = int(results.xyxy[0][i][5])
                    label = labels[label_id]
                    if label == 'nail-biting':
                        print(f'Detected label: {label}, Confidence: {confidence:.2f}')
                        show_fullscreen_message_box()

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualize)
        else:
            lblVideo.image = ""
            cap.release()

def finalize():
    global cap
    cap.release()

def show_fullscreen_message_box():
    messagebox.showinfo("Nail biting detected","Please do not bite your nails!")

cap = None
root = Tk()
root.title("Nail Biting Detection App")
btnIniciar = Button(root, text="Start", width=45, command=inicial)
btnIniciar.grid(column=0, row=0, padx=5, pady=5)
btnFinalizar = Button(root, text="Stop", width=45, command=finalize)
btnFinalizar.grid(column=1, row=0, padx=5, pady=5)
lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)

root.mainloop()