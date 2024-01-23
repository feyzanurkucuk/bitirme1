from tkinter import END, Button, Label, Scale, messagebox, Toplevel, Scrollbar, Listbox
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
                sensitivity = confidence_threshold_var.get()
                sensitivity = (100-sensitivity)/100
                
                if confidence > sensitivity:
                    label_id = int(results.xyxy[0][i][5])
                    label = labels[label_id]
                    
                    
                    if label == 'nail-biting' and time_since_last_detection(last_detection_time) > check_interval_var.get():  # Adjust the delay (in milliseconds)
                        print(f'Detected label: {label}, Confidence: {confidence:.2f}')
                        print(time_since_last_detection(last_detection_time))
                        last_detection_time = time.time()
                        print(last_detection_time)
                        last_detections.append((last_detection_time, label,confidence))
                        if len(last_detections) > 5:
                            last_detections = last_detections[-5:]
                        show_custom_message_box("Nail-biting detected",warning_duration_var.get())
                        #show_fullscreen_message_box()
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


def show_custom_message_box(message,duration):
    fullscreen_warning = Toplevel()
    fullscreen_warning.attributes('-fullscreen', True)  # Set the window to fullscreen
    fullscreen_warning.configure(bg='red')  # Set background color

    # Add a label with the warning message
    warning_label = Label(fullscreen_warning, text=message, font=("Helvetica", 24), fg="white", bg="red")
    warning_label.pack(expand=True)

    # Schedule the window to be destroyed after 'duration' milliseconds
    fullscreen_warning.after(duration*1000, fullscreen_warning.destroy)

    # Bind a function to close the window when any key is pressed
    fullscreen_warning.bind("<Key>", lambda event: fullscreen_warning.destroy())

    # Focus on the window
    fullscreen_warning.focus_force()

    # Make the window transient and grab the focus
    #fullscreen_warning.transient(root)
    fullscreen_warning.grab_set()


def show_fullscreen_message_box():
    messagebox.showwarning("Nail biting detected","Please do not bite your nails!")


def time_since_last_detection(last_detection_time):
    current_time = time.time()
    return current_time - last_detection_time

def update_detections_label():
    lblDetections.delete(0, END)  # Clear the listbox
    for timestamp, label,confidence in last_detections:
        lblDetections.insert(END, f"{label} at {format_timestamp(timestamp)} with Confidence: {confidence:.2f}")

def format_timestamp(timestamp):
    return time.strftime("%H:%M:%S", time.localtime(timestamp))



# Function to update the confidence threshold
def update_confidence_threshold(value):
    confidence_scale_val.configure(text="Sensitivity: " + str(value))


def update_check_interval(value):
    check_interval_val.configure(text="Check Interval: " + str(value) + " s")

def update_warning_duration(value):
    warning_duration_scale.configure(text = "Warning duration:" + str(value))


cap = None
customtkinter.set_appearance_mode("dark")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")
root = customtkinter.CTk()
root.geometry("720x640")
root.title("Nail Biting Detection App")

confidence_threshold_var = tkinter.IntVar(value=30)
check_interval_var = tkinter.IntVar(value=5) #initial
warning_duration_var = tkinter.IntVar(value = 2)
# Create a container frame for video-related widgets
video_container = customtkinter.CTkFrame(root, width=490, height=370)
video_container.grid(column=1, row=1, columnspan=3, padx=10, pady=10, sticky="w")

empty_image = Image.new("RGB", (490, 370), "gray")
empty_image_tk = ImageTk.PhotoImage(empty_image)

# Create a custom label for the video (assuming CTkLabel or a compatible widget)
lblVideo = customtkinter.CTkLabel(video_container,text='',image=empty_image_tk)
lblVideo.grid(column=1, row=1,columnspan=3)

btnIniciar = customtkinter.CTkButton(root, text="Start", width=45, command=inicial)
#btnFinalizar = customtkinter.CTkButton(root, text="Stop", width=45, command=finalize)
btnIniciar.grid(row=1, column=3, padx=10, pady=10)  # "ew" makes the button expand horizontally
#btnFinalizar.grid(column=1, row=0, padx=5, pady=5, sticky="ew")


confidence_scale_val = customtkinter.CTkLabel(master=root,text="Sensitivity: " +str(confidence_threshold_var.get()))
confidence_scale_val.grid(row=2, column=3, sticky="nsew")

confidence_scale = customtkinter.CTkSlider(master=root, from_=10, to=40, number_of_steps=3, command=update_confidence_threshold, variable=confidence_threshold_var)
confidence_scale.grid(row=3, column=3, padx=10, pady=10, sticky="w")

warning_duration_scale = customtkinter.CTkLabel(master=root,text="Warning duration: " +str(warning_duration_var.get()))
warning_duration_scale.grid(row=6, column=3, padx=10, pady=10, sticky="nsew")

check_interval_val = customtkinter.CTkLabel(master=root,text="Check Interval: " +str(check_interval_var.get())+" s")
check_interval_val.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")

check_interval = customtkinter.CTkSlider(master=root, from_=2, to=10,number_of_steps=8, command=update_check_interval, variable=check_interval_var)
check_interval.grid(row=5, column=3, padx=10, pady=10, sticky="w")

warning_duration = customtkinter.CTkSlider(master=root, from_=2, to=7,number_of_steps=5, command=update_warning_duration, variable=warning_duration_var)
warning_duration.grid(row=7, column=3, padx=10, pady=10, sticky="w")



# Create a label for the listbox
lblDetectionsTitle = Label(root, text="Previous Detections", font=("Helvetica", 12, "bold"),relief="solid",bg="gray")
lblDetectionsTitle.grid(column=1, row=2, padx=10, pady=10, sticky="w")

# Create a listbox for previous detections
lblDetections = Listbox(root, height=5, width=100, selectmode="single", exportselection=False,bg="grey")
lblDetections.grid(column=1, row=3, padx=10, pady=10, sticky="w",columnspan=2)




root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)


root.mainloop()