from ultralytics import YOLO
import cv2
import numpy as np
import json

WILDTRACK_PATH = "/Users/valdasv/Downloads/Wildtrack"
VIDEO_PATH1 = f"{WILDTRACK_PATH}/cam1.mp4"
VIDEO_PATH2 = f"{WILDTRACK_PATH}/cam2.mp4"


def lock_and_track_object(video_paths, model_path, output_path, log_path, duration=5):
    model = YOLO(model_path)
    json_log = [] 

    tracker = None
    target_bbox = None
    locked = False
    current_id = 0
    out_of_sight_frames = 0

    def log_position(frame_number, bbox, video_name, object_id):
        entry = {
            "video": video_name,
            "frame_number": frame_number,
            "object_id": object_id,
            "bbox": {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3])
            }
        }
        json_log.append(entry)

    def initialize_tracker(frame, bbox):
        nonlocal tracker
        tracker = cv2.TrackerCSRT_create()
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        tracker.init(frame, (x, y, w, h))

    def is_out_of_sight(bbox, frame_width, frame_height):
        """Determine if the person is out of sight."""
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1

        return (
            box_width < 50 or box_height < 50 or  # Box too small
            x1 < 10 or y1 < 10 or  # Close to left/top edge
            x2 > frame_width - 10 or y2 > frame_height - 10  # Close to right/bottom edge
        )

    def process_frame(frame, frame_number, video_name):
        nonlocal target_bbox, locked, tracker, current_id, out_of_sight_frames

        frame_height, frame_width = frame.shape[:2]

        if locked and tracker is not None:
            # Use the tracker to update the position
            success, box = tracker.update(frame)
            if success:
                x1, y1, w, h = map(int, box)
                target_bbox = [x1, y1, x1 + w, y1 + h]
                log_position(frame_number, target_bbox, video_name, current_id)

                if is_out_of_sight(target_bbox, frame_width, frame_height):
                    out_of_sight_frames += 1
                    if out_of_sight_frames > 10:
                        locked = False
                        tracker = None
                        out_of_sight_frames = 0
                else:
                    out_of_sight_frames = 0
            else:
                locked = False
                tracker = None
                out_of_sight_frames = 0
        else:
            # Perform detection
            results = model(frame)
            detections = results[0].boxes

            if detections is not None and len(detections.xyxy) > 0:
                for detection, cls, conf in zip(
                    detections.xyxy, detections.cls.cpu().numpy(), detections.conf.cpu().numpy()
                ):
                    if int(cls) == 0:  # Check if it's a person
                        bbox = detection.cpu().numpy()

                        # If no match found, initialize a new tracker
                        target_bbox = bbox
                        initialize_tracker(frame, bbox)
                        locked = True
                        current_id += 1  # Assign a new ID

                        log_position(frame_number, target_bbox, video_name, current_id)
                        break

        # Draw the bounding box
        if target_bbox is not None:
            x1, y1, x2, y2 = map(int, target_bbox)
            color = (0, 255, 0)  # Default green for new objects
            label = f"Person ID: {current_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    cap = cv2.VideoCapture(video_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        video_name = video_path.split("/")[-1]
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps
        frame_limit = int(duration * fps) if total_duration > duration else frame_count
        frame_number = 0

        while cap.isOpened() and frame_number < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, frame_number, video_name)
            out.write(frame)
            cv2.imshow('Object Lock and Track', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_number += 1

        cap.release()

    out.release()
    cv2.destroyAllWindows()

    with open(log_path, 'w') as log_file:
        json.dump(json_log, log_file, indent=4)

lock_and_track_object(
    video_paths=[VIDEO_PATH1, VIDEO_PATH2],
    model_path='yolo11n.pt',
    output_path='output/locked_tracking_output.mp4',
    log_path='output/object_movement_log.json'
)