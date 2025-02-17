from tkinter import *
from tkinter.ttk import *
import cv2
from ultralytics import YOLO, solutions
import threading
from PIL import Image, ImageTk

'''
in this code we're attempting to merge counting the object with the gui
this script is crashing :(

'''
# Initialize YOLO model
model = YOLO("weights/yolov10s.pt")  # trained on coco

# Define region points for counting objects
region_points = [(50, 300), (590, 300), (590, 250), (50, 250)] 

# Initialize object counter
counter = solutions.ObjectCounter(
    show=False,  # Change to False to prevent OpenCV window issues
    region=region_points,  
    model="weights/yolov10s.pt",  
    show_in=True,  
    show_out=True  
)


def Initialize_system():
    init = Tk()
    init.geometry('800x700')
    init.title('System')

    # Add a canvas to display the video feed
    canvas = Canvas(init, width=640, height=480, bg="black")
    canvas.pack(pady=20)

    # Label to display object count
    count_label = Label(init, text="Objects Counted: 0", font=("Arial", 12))
    count_label.pack(pady=10)

    def update_video():
        video_source = 'media/demo2.MP4'
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            count_label.config(text="Error: Unable to open video source.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Video ended or error capturing frame.")
                break

            # Count objects in the defined region
            frame = counter.count(frame)
            count_label.config(text=f"Objects Counted: {counter.total}")

            # Convert frame for Tkinter using Pillow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (640, 480))
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update canvas
            canvas.create_image(0, 0, anchor=NW, image=img_tk)
            canvas.image = img_tk  # Keep reference to prevent garbage collection

            init.update()

        cap.release()

    def back():
        init.destroy()
        main()

    back_button = Button(init, text='Back to Menu', command=back)
    back_button.pack(pady=10)

    threading.Thread(target=update_video, daemon=True).start()
    init.mainloop()

def main():
    main = Tk()
    main.geometry('400x300')
    main.title('DeltaVisionX')

    initialize_button = Button(main, text='Initialize System', command=Initialize_system)
    initialize_button.pack(pady=30)

    main.mainloop()

main()
