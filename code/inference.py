import os
import argparse
import pandas as pd
import cv2
import supervision as sv
from ultralytics import YOLO


def process_video(video_path, output_video_path, csv_path, model, img_size, score_th, IOU=0.4):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tracker
    tracker = sv.ByteTrack(frame_rate=fps)

    # Writer
    writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    # Annotators
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    csv_log = []
    permanent_traces_list = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=score_th, iou=IOU, imgsz=img_size, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update_with_detections(detections)

        # Default logging = not visible
        cx, cy, visibility = -1, -1, 0

        if len(tracked) > 0:
            x1, y1, x2, y2 = tracked.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            permanent_traces_list.append((cx,cy))
            visibility = 1

        # Write annotations for CSV
        csv_log.append((frame_id, cx, cy, visibility))

        # Draw annotations
        labels = []
        for i in range(len(tracked)):
            cid = tracked.class_id[i]
            conf = tracked.confidence[i]
            tid = tracked.tracker_id[i]
            class_name = results.names.get(cid, str(cid))
            labels.append(f"{class_name}")

        frame = label_annotator.annotate(scene=frame, detections=tracked, labels=labels)

        for j in range(1, len(permanent_traces_list)):
            cv2.line(frame, permanent_traces_list[j-1], permanent_traces_list[j], (0, 255, 0), 2)  # green line
        
        if visibility ==1:
            cv2.circle(frame, (cx, cy), radius=4, color=(0,0,255), thickness=-1)

        writer.write(frame)
        frame_id += 1
    
    csv_log_df = pd.DataFrame(csv_log, columns=["frame", "centroid_x", "centroid_y", "visible"])
    csv_log_df.to_csv(csv_path, index=False, header=False)
    cap.release()
    writer.release()


def process_directory(input_dir, output_dir, logs_dir, model_path, img_size, score_thresh):

    # Load YOLO model once
    model = YOLO(model_path)

    # Create output directories if missing
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):

            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_processed.mp4")
            csv_path = os.path.join(logs_dir, f"{os.path.splitext(file)[0]}.csv")

            print(f"Processing -> {file}")

            process_video(
                video_path=input_path,
                output_video_path=output_path,
                csv_path=csv_path,
                model=model,
                img_size=img_size,
                score_th=score_thresh
            )

    print("Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="Directory containing input videos")
    parser.add_argument("--output_dir", required=True, help="Directory to save processed videos")
    parser.add_argument("--logs_dir", required=True, help="Directory to save CSV log files")
    parser.add_argument("--model", required=True, help="YOLO model path (.pt)")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--score_th", type=float, default=0.5, help="Detection score threshold")

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        logs_dir=args.logs_dir,
        model_path=args.model,
        img_size=args.imgsz,
        score_thresh = args.score_th
    )
