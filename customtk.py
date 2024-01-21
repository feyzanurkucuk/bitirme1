from tkinter import END, Button, Label, messagebox, Toplevel, Scrollbar, Listbox
import tkinter
import cv2
from tkinter import Tk
from PIL import Image, ImageTk
import torch
import numpy as np
import pathlib
import customtkinter
import time




last_detections = []
last_detection_time = 0

pathlib.PosixPath = pathlib.WindowsPath
# Load a pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
model = torch.hub.load('ultralytics/yolov5','custom',path= 'weights_last/best.pt', force_reload=True)

def inicial():
    global cap
    if btnIniciar.cget("text") == "Start":
        print("Starting prediction...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        visualize()
    else:
        print("Stopping prediction...")
        cap.release()
    btnIniciar.configure(text = "Stop" if btnIniciar.cget("text") == "Start" else "Start")


def visualize():
    
    global cap ,last_detections, last_detection_time
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
                    
                    if label == 'nail-biting' and time_since_last_detection(last_detection_time) > 5:  # Adjust the delay (in milliseconds)
                        print(f'Detected label: {label}, Confidence: {confidence:.2f}')
                        print(time_since_last_detection(last_detection_time))
                        last_detection_time = time.time()
                        print(last_detection_time)
                        last_detections.append((last_detection_time, label))
                        if len(last_detections) > 5:
                            last_detections = last_detections[-5:]
                        show_fullscreen_message_box()
                        update_detections_label()

            frame = cv2.resize(frame, (480, 360))
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
    messagebox.showwarning("Nail biting detected","Please do not bite your nails!")


def time_since_last_detection(last_detection_time):
    current_time = time.time()
    return current_time - last_detection_time

def update_detections_label():
    lblDetections.delete(0, END)  # Clear the listbox
    for timestamp, label in last_detections:
        lblDetections.insert(END, f"{label} at {format_timestamp(timestamp)}")

def format_timestamp(timestamp):
    return time.strftime("%H:%M:%S", time.localtime(timestamp))

cap = None
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()
root.geometry("720x640")
root.title("Nail Biting Detection App")


# Create a container frame for video-related widgets
video_container = customtkinter.CTkFrame(root, width=490, height=370)
video_container.grid(column=0, row=1, columnspan=2, padx=10, pady=10, sticky="w")

empty_image = Image.new("RGB", (490, 370), "gray")
empty_image_tk = ImageTk.PhotoImage(empty_image)

# Create a custom label for the video (assuming CTkLabel or a compatible widget)
lblVideo = customtkinter.CTkLabel(video_container,text='',image=empty_image_tk)
lblVideo.grid(column=0, row=0)

btnIniciar = customtkinter.CTkButton(root, text="Start", width=45, command=inicial)
#btnFinalizar = customtkinter.CTkButton(root, text="Stop", width=45, command=finalize)
btnIniciar.grid(column=0, row=0, padx=10, pady=10)  # "ew" makes the button expand horizontally
#btnFinalizar.grid(column=1, row=0, padx=5, pady=5, sticky="ew")


# Create a label for the listbox
lblDetectionsTitle = Label(root, text="Previous Detections", font=("Helvetica", 12, "bold"),relief="solid",bg="gray")
lblDetectionsTitle.grid(column=2, row=0, padx=10, pady=10, sticky="n")

# Create a listbox for previous detections
lblDetections = Listbox(root, height=5, width=30, selectmode="single", exportselection=False,bg="grey")
lblDetections.grid(column=2, row=1, padx=10, pady=10, sticky="n")

# Add a vertical scrollbar to the listbox
scrollbar = Scrollbar(root, orient="vertical")
scrollbar.grid(column=3, row=1, sticky="nsw")
lblDetections.config(yscrollcommand=scrollbar.set)
scrollbar.config(command=lblDetections.yview)


root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)


root.mainloop()