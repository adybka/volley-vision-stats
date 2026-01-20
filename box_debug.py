import argparse
import os

import cv2
from handlers.serving import (
    box_sizing_serve_detection,
    horizontal_split_serve_detection,
)
import torch
from ultralytics import YOLO

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


def main(arg):
    model_path = args.model
    video_path = args.input_path
    output_path = args.output_path

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
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties for saving the output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:", fps)
    #make dir if output path dir doesn't exist
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    serve_count = {0: 0, 1: 0}  # Dictionary to count serves for each team

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(
            frame,
            conf=args.conf,
            max_det=args.max_det,
            classes=args.classes,
            verbose=False,
        )

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == SERVE_CLASS_ID:
                    # Determine team using selected method
                    if args.team_select == 0:
                        team_label, color = (
                            horizontal_split_serve_detection.horizontal_split_serve_detection(
                                box.xyxy[0], height
                            )
                        )
                    elif args.team_select == 1:
                        team_label, color = (
                            box_sizing_serve_detection.horizontal_split_serve_detection(
                                box.xyxy[0], height
                            )
                        )
                    serve_count[team_label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Draw Label
                    label = f"Serve: {team_label}"
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(
                        frame, (x1, y1 - t_size[1] - 5), (x1 + t_size[0], y1), color, -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete.")
    print(f"Serve counts: {serve_count}")


main(args)
