from ultralytics import YOLO
import cv2
import numpy as np
import json
from torchvision import transforms
from torchvision.models import resnet50
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import torch

WILDTRACK_PATH = "/Users/valdasv/Downloads/Wildtrack"
VIDEO_PATH1 = f"{WILDTRACK_PATH}/cam1.mp4"
VIDEO_PATH2 = f"{WILDTRACK_PATH}/cam4.mp4"


def setup_reid_model():
    extractor = FeatureExtractor(
        model_name='osnet_x1_0',
        model_path='',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return extractor


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
    x1, y1, x2, y2 = map(int, bbox)
    cropped = frame[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    embedding = reid_model(cropped)
    return embedding.flatten()


def match_with_cache(embedding, cache):
    if not cache:
        return None
    similarities = [cosine_similarity([embedding], [obj["embedding"]])[0][0] for obj in cache]
    best_match_index = int(np.argmax(similarities))
    best_similarity = similarities[best_match_index]
    if best_similarity > 0.8:
        return cache[best_match_index]["id"]
    return None


def lock_and_track_object(video_paths, model_path, reid_model_path, output_path, log_path, duration=5):
    model = YOLO(model_path)
    reid_model = setup_reid_model()
    json_log = []
    cache = []
    tracked_objects = []
    current_id = 0

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

    def process_frame(frame, frame_number, video_name):
        nonlocal cache, tracked_objects, current_id
        results = model(frame)
        detections = results[0].boxes

        if detections is not None and len(detections.xyxy) > 0:
            for detection, cls in zip(detections.xyxy, detections.cls.cpu().numpy()):
                if int(cls) == 0:  # Person class
                    bbox = detection.cpu().numpy()
                    embedding = extract_embedding(reid_model, frame, bbox)

                    matched_id = match_with_cache(embedding, cache)
                    if matched_id is not None:
                        log_position(frame_number, bbox, video_name, matched_id, from_cache=True)
                        color = (0, 0, 255)
                        label = f"FROM CACHE: ID {matched_id}"
                        if matched_id not in tracked_objects:
                            tracked_objects.append(matched_id)
                    else:
                        if len(tracked_objects) < 5:  # Limit the number of tracked objects to 5
                            current_id += 1

                            if len(cache) >= 5:
                                cache.pop(0)

                            cache.append({"id": current_id, "embedding": embedding})
                            tracked_objects.append(current_id)
                            log_position(frame_number, bbox, video_name, current_id)
                            color = (0, 255, 0)
                            label = f"New ID: {current_id}"
                        else:
                            continue

                    x1, y1, x2, y2 = map(int, bbox)
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
