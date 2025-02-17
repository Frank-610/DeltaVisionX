from tkinter import *
from tkinter.ttk import *
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk

'''
in this script we are attempting to create a gui where the user chooses the yolo model so that the program detects the object classes accordingly
no threading, no queue
'''
# Initialize YOLO model (default for cell phones)
model = YOLO("weights/yolov10s.pt")  # Trained on COCO



def Initialize_system():
    init = Tk()
    init.geometry('800x700')
    init.title('System')

    # Add a canvas to display the video feed
    canvas = Canvas(init, width=640, height=480, bg="black")
    canvas.pack(pady=20)

    # Label to display coordinates
    coord_label = Label(init, text="Detected Coordinates: None", font=("Arial", 12))
    coord_label.pack(pady=10)

    def update_video():
        # Open video feed
        video_source = 'media/demo2.MP4' #this is a video that simulates phones moving on a conveyor belt
        cap = cv2.VideoCapture(video_source)
        #cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            coord_label.config(text="Error: Unable to open video source.")
            return

        def process_frame():
            ret, frame = cap.read()
            if not ret:
                coord_label.config(text="Video ended or error capturing frame.")
                cap.release()
                return

            # Run YOLO inference (detect class 67 - cell phone)
            results = model.predict(frame, conf=0.5, classes=[67])

            # Draw bounding boxes and update coordinates
            coords = []
            for result in results:
                boxes = result.boxes  # Bounding boxes
                for box in boxes:
                    if box.xyxy is not None and box.conf is not None:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
                        cv2.putText(frame, f"Conf: {box.conf[0]:.2f}",
                                    (x1, y1 - 10),  # Position the text above the box
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add confidence text
                        coords.append((x1, y1))  # Store coordinates

            # Update coordinates label
            coord_label.config(text=f"Detected Coordinates: {coords}" if coords else "Detected Coordinates: None")

            # Convert frame for Tkinter compatibility
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))

            # Convert to a PIL image
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update canvas
            canvas.create_image(0, 0, anchor=NW, image=img_tk)
            canvas.image = img_tk  # Keep a reference to prevent garbage collection

            # Repeat after a short delay to maintain frame rate
            init.after(2, process_frame)

        process_frame()

    def back():
        init.destroy()
        main()

    back_button = Button(init, text='Back to Menu', command=back)
    back_button.pack(pady=10)

    update_video()  # Start the video feed update loop

    init.mainloop()

def create():
    createDataset = Tk()
    createDataset.geometry('800x700')
    createDataset.title('System')

    def back():
        createDataset.destroy()
        main()

    back_button = Button(createDataset, text='Back to Menu', command=back)
    back_button.pack(pady=10)

    createDataset.mainloop()

def main():
    main = Tk()
    main.geometry('400x300')
    main.title('DeltaVisionX')

    select = Label(main, text='Select class:')
    select.pack(pady=10)

    which_model = Combobox(main)
    which_model['values'] = ['Cell Phones', 'Faces']
    which_model.current(0)
    which_model.pack(pady=5)

    def initialize():
        global model
        if which_model.current() == 1:
            model = YOLO("weights/yolov11n-face.pt")  # Load face detection model
        else:
            model = YOLO("weights/yolov11n.pt")  # Load cell phone detection model
        main.destroy()
        Initialize_system()

    initialize_button = Button(main, text='Initialize System', command=initialize)
    initialize_button.pack(pady=30)

    def create_dataset():
        main.destroy()
        create()

    dataset_button = Button(main, text='Create dataset', command=create_dataset)
    dataset_button.pack(pady=30)

    main.mainloop()

main()
