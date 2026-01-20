import argparse
import os
import cv2
from handlers.serving import (
    box_sizing_serve_detection,
    horizontal_split_serve_detection,
)
import torch
from ultralytics import YOLO
from collections import deque, Counter

parser = argparse.ArgumentParser(description="YOLOv8 Image/Video Processing")
parser.add_argument("--model", required=True, help="Path to model's weights")
parser.add_argument(
    "--input_path", required=True, help="Path to the input image or video file"
)
parser.add_argument(
    "--output_path",
    default="Output/output.jpg",
    help="Output directory path (for images) or output file path (for videos)",
)
parser.add_argument(
    "--show_conf",
    default=False,
    action="store_true",
    help="Whether to show the confidence scores",
)
parser.add_argument(
    "--show_labels",
    default=False,
    action="store_true",
    help="Whether to show the labels",
)
parser.add_argument(
    "--conf", type=float, default=0.5, help="Object confidence threshold for detection"
)
parser.add_argument(
    "--max_det", type=int, default=300, help="Maximum number of detections per image"
)
parser.add_argument(
    "--classes", nargs="+", default=None, help="List of classes to detect"
)
parser.add_argument(
    "--line_width",
    type=int,
    default=3,
    help="Line width for bounding box visualization",
)
parser.add_argument(
    "--font_size", type=float, default=3, help="Font size for label visualization"
)
parser.add_argument(
    "--gpu", default=False, action="store_true", help="Whether to use GPU"
)
parser.add_argument(
    "--team_select",
    default=0,
    type=int,
    help="How to select teams: 0=horizontal-split, 1=box-sizing",
)
args = parser.parse_args()


def main(args):
    model_path = args.model
    video_path = args.input_path
    output_path = args.output_path
    classes = {0: "block", 1: "defense", 2: "serve", 3: "set", 4: "spike"}

    if args.gpu:
        if torch.cuda.is_available():
            print("Using GPU for inference.")
            torch.cuda.set_device(0)
        else:
            print("GPU specified but not available. Using CPU.")
    model = YOLO(model_path)

    # Classes: {0: 'block', 1: 'defense', 2: 'serve', 3: 'set', 4: 'spike'}
    SERVE_CLASS_ID = 2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    event_detected = deque(maxlen=15)  # what to display for the next 15 frames
    sliding_window = deque(maxlen=5)  # sliding window of 5 frames

    # Event will be declared when
    # the event_counter exceeds 3
    event = False

    # Loop through the video frames
    frame_num = 1
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            results = model.predict(
                frame,
                conf=args.conf,
                max_det=args.max_det,
                classes=args.classes,
                verbose=False,
            )

            annotated_frame = frame.copy()
            font_scale = 2  # 0.8
            thickness = 2  # 1
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Check if there is a detection in the current frame
            if len(results[0]) > 0:
                # class names of all detections
                cls = [idx for idx in results[0].boxes.cls.tolist()]
                if SERVE_CLASS_ID in cls:
                    sliding_window.extend(cls)
                    cls, count = Counter(sliding_window).most_common(1)[0]

                    if (count >= 3) and (cls is not None):
                        event = True
            else:
                sliding_window.append(None)

            # If event detected 3 times, announce
            if event:
                event = False
                for box in results[0].boxes:
                    if int(box.cls) == SERVE_CLASS_ID:
                        print(horizontal_split_serve_detection.horizontal_split_serve_detection(box.xyxy[0], frame.shape[1]))
                        
                for i in range(15):
                    event_detected.append(cls)

            try:
                # Draw the event detected at the bottom of the frame
                event_msg = f"{classes[event_detected.popleft()]}".upper()
                text_size, _ = cv2.getTextSize(event_msg, font, font_scale, thickness)
                text_x = (annotated_frame.shape[1] - text_size[0]) // 2
                text_y = 0
                cv2.rectangle(
                    annotated_frame,
                    (text_x - 20, text_y),
                    (text_x + text_size[0] + 150, text_y + text_size[1] + 80),
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    event_msg,
                    (text_x, text_y + 80),
                    font,
                    3,
                    (255, 255, 255),
                    2,
                )
            except IndexError:
                pass

            # Write the annotated frame to the output video
            out.write(annotated_frame)
            frame_num += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object, close the display window, and release the output video writer
    cap.release()
    cv2.destroyAllWindows()
    out.release()


main(args)
