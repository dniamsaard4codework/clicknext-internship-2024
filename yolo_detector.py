from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2

# Create a list to keep the previous center point for tracking line
previous_point = []

# Load YOLO model
model = YOLO("yolov8n.pt")

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""
    global class_id, class_name, coordinator, confidence

    # Create annotator object
    annotator = Annotator(frame)
    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        confidence = box.conf

    # Create blue-bounding-box on cat detection
    if class_name == 'cat':
        bounding_color = (255, 0, 0)
    else:
        bounding_color = colors(class_id, True)

    # Draw bounding box on only cat
    annotator.box_label(
        box=coordinator, label=class_name, color=bounding_color
    )

    # Set the variables for all values in coordinator
    x1,y1,x2,y2 = coordinator

    # Find the center of the box to make the plotting point
    track_center = ((x1+x2)/2,(y1+y2)/2)
    previous_point.append(track_center)

    # Draw the tracking line for whole cat detection
    for i in range(1, len(previous_point)):
        start = (int(previous_point[i-1][0]), int(previous_point[i-1][1]))
        end = (int(previous_point[i][0]), int(previous_point[i][1]))
        cv2.line(frame, start, end, (81,255,81), 3)

    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame (Detect only "cat" class)
    results = model(frame, classes = 15)

    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect motorcycle from image frame
            frame_result = detect_object(frame)
            # Show result
            cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
            # Write text on the top-right corner of the video
            cv2.putText(frame,  
                'Dechathon-Clicknext-Internship-2024',  
                (frame.shape[0]-60, 30),  
                cv2.FONT_HERSHEY_SIMPLEX , 1,  
                (0, 0, 255),  
                2,  
                cv2.LINE_4) 
            cv2.imshow("Video", frame_result)
            cv2.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()
