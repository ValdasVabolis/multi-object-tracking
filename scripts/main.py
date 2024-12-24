from ultralytics import YOLO
import cv2
import numpy as np
import json
from torchvision import transforms
from torchvision.models import resnet50
from sklearn.metrics.pairwise import cosine_similarity
import torch

WILDTRACK_PATH = "/Users/valdasv/Downloads/Wildtrack"
VIDEO_PATH1 = f"{WILDTRACK_PATH}/cam1.mp4"
VIDEO_PATH2 = f"{WILDTRACK_PATH}/cam4.mp4"


def setup_reid_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")

    reid_model = resnet50()
    reid_model.fc = torch.nn.Linear(reid_model.fc.in_features, 512)
    reid_model.load_state_dict(checkpoint, strict=False)

    reid_model.eval()
    reid_model.to("cuda" if torch.cuda.is_available() else "cpu")
    return reid_model


def preprocess_reid_image(frame, bbox):
    # Ensure bounding box is within frame dimensions
    height, width, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    # Crop and preprocess
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:  # Handle empty crops gracefully
        raise ValueError("Invalid crop dimensions: bbox is outside frame bounds.")
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(cropped).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")



def extract_embedding(reid_model, frame, bbox):
    tensor = preprocess_reid_image(frame, bbox)
    with torch.no_grad():
        embedding = reid_model(tensor)
    return embedding.cpu().numpy().flatten()


def match_with_cache(embedding, cache):
    if not cache:
        return None
    similarities = [cosine_similarity([embedding], [obj["embedding"]])[0][0] for obj in cache]
    best_match_index = int(np.argmax(similarities))
    best_similarity = similarities[best_match_index]
    if best_similarity > 0.8:  # Threshold for matching
        return cache[best_match_index]["id"]
    return None


def lock_and_track_object(video_paths, model_path, reid_model_path, output_path, log_path, duration=5):
    model = YOLO(model_path)
    reid_model = setup_reid_model(reid_model_path)
    json_log = []  # To log object movement
    tracker = None
    target_bbox = None
    locked = False
    current_id = 0
    out_of_sight_frames = 0
    cache = []

    def log_position(frame_number, bbox, video_name, object_id, from_cache=False):
        entry = {
            "video": video_name,
            "frame_number": frame_number,
            "object_id": object_id,
            "from_cache": from_cache,
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
        x1, y1, x2, y2 = bbox
        box_width, box_height = x2 - x1, y2 - y1
        return (
            box_width < 50 or box_height < 50 or
            x1 < 10 or y1 < 10 or
            x2 > frame_width - 10 or y2 > frame_height - 10
        )

    def process_frame(frame, frame_number, video_name):
        nonlocal target_bbox, locked, tracker, current_id, out_of_sight_frames
        frame_height, frame_width = frame.shape[:2]

        cache_info = [f"ID {obj['id']}: {'Present' if obj.get('present', False) else 'Not Present'}" for obj in cache]
        for idx, text in enumerate(cache_info):
            cv2.putText(frame, text, (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if locked and tracker is not None:
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
            results = model(frame)
            detections = results[0].boxes

            if detections is not None and len(detections.xyxy) > 0:
                for detection, cls, conf in zip(
                    detections.xyxy, detections.cls.cpu().numpy(), detections.conf.cpu().numpy()
                ):
                    if int(cls) == 0:
                        bbox = detection.cpu().numpy()
                        embedding = extract_embedding(reid_model, frame, bbox)

                        matched_id = match_with_cache(embedding, cache)
                        if matched_id:
                            target_bbox = bbox
                            initialize_tracker(frame, bbox)
                            locked = True
                            log_position(frame_number, target_bbox, video_name, matched_id, from_cache=True)
                            return frame

                        target_bbox = bbox
                        initialize_tracker(frame, bbox)
                        locked = True
                        current_id += 1

                        if len(cache) < 3:
                            cache.append({"id": current_id, "embedding": embedding, "bbox": bbox})

                        log_position(frame_number, target_bbox, video_name, current_id)
                        break

        if target_bbox is not None:
            x1, y1, x2, y2 = map(int, target_bbox)
            color = (0, 255, 0)
            label = f"Person ID: {current_id}"

            if match_with_cache(extract_embedding(reid_model, frame, target_bbox), cache):
                color = (0, 0, 255)
                label = "FROM CACHE:"

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
    model_path='../models/yolo11n.pt',
    reid_model_path='../models/msmt_sbs_S50.pth',
    output_path='output/locked_tracking_output.mp4',
    log_path='output/object_movement_log.json'
)
