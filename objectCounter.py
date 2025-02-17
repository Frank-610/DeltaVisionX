import cv2
from ultralytics import solutions

'''
this script counts mobile phones using ultralytics' solutions library
'''
# Open video file
cap = cv2.VideoCapture("media/demo2.mp4")
cap.set(cv2.CAP_PROP_FPS, 4)
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(w // 2, 10), (w // 2, h - 10)]  # Centered vertical line

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Let ObjectCounter display the counting overlay
    region=region_points,
    model="weights/yolov11n.pt",
    classes=[67],  # Counting a specific class (e.g., cell phone)
)

# Video writer (match input format)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("object_counting_output.mp4", fourcc, fps, (w, h))

# Process video
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing completed.")
        break

    # Count objects
    frame = counter.count(frame)

    # Show the frame
    #cv2.imshow("Object Counting", frame)

    # Write frame to output video
    video_writer.write(frame)

    # Maintain normal speed using frame rate delay
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
video_writer.release()
cv2.destroyAllWindows()
