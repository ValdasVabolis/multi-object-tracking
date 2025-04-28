from ultralytics import YOLO
import cv2
import numpy as np
import json
from torchvision import transforms
from torchvision.models import resnet50
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import torch
import time

WILDTRACK_PATH = "/Users/valdasv/Downloads/Wildtrack"
VIDEO_PATH1 = f"{WILDTRACK_PATH}/cam1_10.mp4"
VIDEO_PATH2 = f"{WILDTRACK_PATH}/cam4_10.mp4"
VIDEO_PATH3 = f"{WILDTRACK_PATH}/cam6_10.mp4"

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

def calculate_combined_similarity(embedding, cache_embeddings):
    if not cache_embeddings:
        return None, 0
    cache_matrix = np.stack(cache_embeddings)  # Stack all embeddings into a single matrix
    similarities = cosine_similarity([embedding], cache_matrix)[0]
    best_index = np.argmax(similarities)
    return best_index, similarities[best_index]

def match_with_cache(embedding, cache):
    if not cache:
        return None
    similarities = [cosine_similarity([embedding], [obj["embedding"]])[0][0] for obj in cache]
    best_match_index = int(np.argmax(similarities))
    best_similarity = similarities[best_match_index]
    if best_similarity > 0.8:
        return cache[best_match_index]["id"]
    return None

def is_valid_embedding(new_embedding, existing_embeddings, threshold=0.7):
    if not existing_embeddings:
        return True
    similarities = [cosine_similarity([new_embedding], [e])[0][0] for e in existing_embeddings]
    average_similarity = np.mean(similarities)
    return average_similarity >= threshold

def validate_cache(cache, threshold=0.7):
    for obj in cache:
        embeddings = np.array(obj['embeddings'])
        if len(embeddings) > 1:
            similarities = cosine_similarity(embeddings)
            median_similarity = np.median(similarities)
            if median_similarity < threshold:
                obj['embeddings'] = list(embeddings[np.where(similarities >= threshold)])
                obj['previews'] = obj['previews'][:len(obj['embeddings'])]

def lock_and_track_object(video_paths, model_path, output_path, log_path, duration=10):
    model = YOLO(model_path)
    reid_model = setup_reid_model()
    json_log = []
    cache = []
    tracked_objects = []
    current_id = 0
    id_switch_count = 0
    last_tracked_ids = {}
    flash_cache_ids = set()
    
    total_frames = 0
    matched_frames = 0

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

    
    def display_cache(frame):
        x_offset, y_offset = 10, 10
        frame_height, frame_width, _ = frame.shape

        for obj in cache[:5]:  # Limit to 5 people
            label = f"ID#{obj['id']} (Frames: {len(obj['matched_frames'])})"
            color = (0, 255, 0) if obj['id'] in flash_cache_ids else (0, 0, 255)
            cv2.putText(frame, label, (x_offset, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Adjust y_offset to create space for label
            preview_y_offset = y_offset + 30

            for i, preview in enumerate(obj['previews'][:10]):  # Limit to 10 previews per person
                x_preview = x_offset + (60 * i)
                resized_preview = cv2.resize(preview, (50, 100))
                if len(resized_preview.shape) == 2:  # Handle grayscale previews
                    resized_preview = cv2.cvtColor(resized_preview, cv2.COLOR_GRAY2BGR)

                if preview_y_offset + 100 <= frame_height and x_preview + 50 <= frame_width:
                    frame[preview_y_offset:preview_y_offset + 100, x_preview:x_preview + 50] = resized_preview

            y_offset += 140  # Move vertically for the next person

    def calculate_combined_similarity(embedding, obj_embeddings):
        similarities = [cosine_similarity([embedding], [e])[0][0] for e in obj_embeddings]
        return max(similarities)

    def process_frame(frame, frame_number, video_name):
        nonlocal cache, tracked_objects, current_id, id_switch_count, last_tracked_ids, flash_cache_ids, matched_frames
        results = model(frame)
        detections = results[0].boxes

        flash_cache_ids.clear()

        if frame_number % 100 == 0:
            validate_cache(cache)

        if detections is not None and len(detections.xyxy) > 0:
            match_found = False
            for detection, cls in zip(detections.xyxy, detections.cls.cpu().numpy()):
                if int(cls) == 0:
                    bbox = detection.cpu().numpy()
                    embedding = extract_embedding(reid_model, frame, bbox)

                    matched_id = None
                    best_similarity = 0

                    for obj in cache:
                        similarity = calculate_combined_similarity(embedding, obj['embeddings'])
                        if similarity > best_similarity and similarity > 0.8:
                            best_similarity = similarity
                            matched_id = obj['id']

                    if matched_id is not None:
                        match_found = True
                        log_position(frame_number, bbox, video_name, matched_id, from_cache=True)
                        color = (0, 0, 255)
                        label = f"FROM CACHE: ID {matched_id}"

                        bbox_key = tuple(map(int, bbox))
                        if bbox_key in last_tracked_ids and last_tracked_ids[bbox_key] != matched_id:
                            id_switch_count += 1
                        last_tracked_ids[bbox_key] = matched_id

                        if matched_id not in tracked_objects:
                            tracked_objects.append(matched_id)

                        flash_cache_ids.add(matched_id)

                        # Update cache with new matched frame
                        for obj in cache:
                            if obj['id'] == matched_id:
                                if not is_valid_embedding(embedding, obj['embeddings']):
                                    break
                                obj['embeddings'].append(embedding)
                                if frame_number % 10 == 0:  # Limit sampling to every 10th frame
                                    cropped = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                                    if cropped.size != 0:
                                        obj['previews'].append(cv2.resize(cropped, (50, 100)))
                                        if len(obj['previews']) > 10:  # Limit previews to 10 per person
                                            obj['previews'].pop(0)
                                obj['matched_frames'].append(frame_number)  # Append matched frame number
                        
                    else:
                        if len(tracked_objects) < 5:
                            current_id += 1
                            if len(cache) >= 5:
                                cache.pop(0)

                            x1, y1, x2, y2 = map(int, bbox)
                            cropped = frame[y1:y2, x1:x2]
                            previews = []
                            if cropped.size != 0:
                                obj_image = cv2.resize(cropped, (50, 100))
                                previews.append(obj_image)

                            cache.append({
                                "id": current_id,
                                "embeddings": [embedding],
                                "previews": previews,
                                "matched_frames": []  # Initialize matched_frames
                            })

                            tracked_objects.append(current_id)
                            log_position(frame_number, bbox, video_name, current_id)
                            color = (0, 255, 0)
                            label = f"New ID: {current_id}"
                        else:
                            continue

                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if match_found:
                matched_frames += 1

        display_cache(frame)
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

        for frame_number in range(frame_limit):
            total_frames += 1
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame, frame_number, video_name)
            out.write(frame)
            cv2.imshow('Object Lock and Track', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    out.release()
    cv2.destroyAllWindows()

    match_ratio = matched_frames / total_frames if total_frames > 0 else 0
    print(f"Total Frames Processed: {total_frames}")
    print(f"Frames with Matches: {matched_frames}")
    print(f"Match Ratio: {match_ratio:.2f}")

lock_and_track_object(
    video_paths=[VIDEO_PATH1, VIDEO_PATH2, VIDEO_PATH3],
    model_path='../models/yolo11n.pt',
    output_path='output/locked_tracking_output.mp4',
    log_path='output/object_movement_log.json'
)