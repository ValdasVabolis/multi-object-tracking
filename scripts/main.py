from ultralytics import YOLO
import cv2
import numpy as np
import json
from torchvision import transforms
from torchvision.models import resnet50
from torchreid.utils import FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity
import torch
import random
import os

from mota_calculator import MOTACalculator

ANNOTATIONS_DIR = '/Users/valdasv/Downloads/Wildtrack/annotations_positions'

WILDTRACK_PATH = "/Users/valdasv/Downloads/Wildtrack"
VIDEO_PATH1 = f"{WILDTRACK_PATH}/cam1.mp4"
VIDEO_PATH2 = f"{WILDTRACK_PATH}/cam4.mp4"
VIDEO_PATH3 = f"{WILDTRACK_PATH}/cam6.mp4"
VIDEO_PATH4 = f"{WILDTRACK_PATH}/cam2.mp4"
VIDEO_PATH5 = f"{WILDTRACK_PATH}/cam3.mp4"
VIDEO_PATH6 = f"{WILDTRACK_PATH}/cam5.mp4"
VIDEO_PATH7 = f"{WILDTRACK_PATH}/cam7.mp4"


MAX_EMBEDDINGS = 50
MAX_PREVIEWS = 10
MAX_MATCHED_FRAMES = 10

id_map = {
    2: 9,
    4: 40,
    3: 8
}

focus_id = {
    2: 9
}

view_map = {
    "cam1.mp4": 0,
    "cam2.mp4": 1,
    "cam3.mp4": 2,
    "cam4.mp4": 3,
    "cam5.mp4": 4,
    "cam6.mp4": 5,
    "cam7.mp4": 6,
}


mota_calc = MOTACalculator(
    annotation_dir=ANNOTATIONS_DIR,
    id_mapping=id_map
)

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

def draw_ground_truth_boxes(frame, frame_number, video_name, annotation_dir, view_map, scale_x = 0.0, scale_y = 0.0):
    view_num = view_map.get(video_name)
    if view_num is None:
        return

    annotation_path = os.path.join(annotation_dir, f"{frame_number:08d}.json")
    if not os.path.exists(annotation_path):
        return

    with open(annotation_path, "r") as f:
        data = json.load(f)

    for person in data:
        for view in person["views"]:
            if view["viewNum"] == view_num:
                if view["xmax"] == -1 or view["xmin"] == -1:
                    continue  # Skip invalid boxes
                x1, y1 = int(view["xmin"] * scale_x), int(view["ymin"] * scale_y)
                x2, y2 = int(view["xmax"] * scale_x), int(view["ymax"] * scale_y)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
                cv2.putText(frame, f"GT:{person['personID']}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

def evaluation_mode(video_path, annotation_image_width, annotation_image_height, valid_ids, process_frame, get_color):
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_name} opened: {video_fps:.2f} FPS, resolution {video_width}x{video_height}")

    # Determine which camera this is (cam1 => viewNum=0, cam2 => viewNum=1, etc.)
    try:
        view_num_target = int(video_name.replace("cam", "").replace(".mp4", "")) - 1
    except ValueError:
        print(f"Could not determine view number from video name: {video_name}")
        return

    scale_x = video_width / annotation_image_width
    scale_y = video_height / annotation_image_height

    frame_idx = 0
    annotation_idx = 0

    while cap.isOpened() and annotation_idx < 2000:
        correct_video_frame_idx = round(annotation_idx * (video_fps / 10))
        cap.set(cv2.CAP_PROP_POS_FRAMES, correct_video_frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("End of video or error.")
            break

        if frame_idx % int(video_fps // 10) == 0:
            annotation_file = os.path.join(ANNOTATIONS_DIR, f"{annotation_idx:08d}.json")
            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    annotations = json.load(f)

                frame = process_frame(frame, correct_video_frame_idx, video_name, scale_x, scale_y)

                for person in annotations:
                    person_id = person["personID"]
                    views = person["views"]
                    if person_id not in valid_ids:
                        continue
                    for view in views:
                        if view["viewNum"] != view_num_target or view["xmin"] == -1:
                            continue

                        xmin = int(view["xmin"] * scale_x)
                        xmax = int(view["xmax"] * scale_x)
                        ymin = int(view["ymin"] * scale_y)
                        ymax = int(view["ymax"] * scale_y)

                        color = get_color(person_id)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, f"GT personID: {person_id}", (xmin, ymin - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.imshow(f'{video_name} with Bounding Boxes', frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break

            annotation_idx += 1

        frame_idx += 1

def presentation_mode(video_path, duration, fps, total_frames, process_frame, out):
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
        frame = process_frame(frame, frame_number, video_name, 1.0, 1.0)
        out.write(frame)
        cv2.imshow('Object Lock and Track', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def lock_and_track_object(video_paths, model_path, output_path, log_path, duration=10, eval_mode=False):
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

            for i, preview in enumerate(obj['previews'][:5]):  # Limit to 10 previews per person
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

    def process_frame(frame, frame_number, video_name, scale_x, scale_y):
        nonlocal cache, tracked_objects, current_id, id_switch_count, last_tracked_ids, flash_cache_ids, matched_frames
        # draw_ground_truth_boxes(frame, frame_number, video_name, ANNOTATIONS_DIR, view_map, scale_x, scale_y)
        results = model(frame)
        detections = results[0].boxes

        flash_cache_ids.clear()


        if frame_number % 50 == 0:
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
                        else:
                            mota_calc.log_prediction(video_name, frame_number, bbox.tolist(), matched_id)
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
                        if len(tracked_objects) < 2:
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
                                "matched_frames": []
                            })

                            tracked_objects.append(current_id)
                            log_position(frame_number, bbox, video_name, current_id)
                            color = (0, 255, 0)
                            label = f"New ID: {current_id}"
                        else:
                            continue

                    x1, y1, x2, y2 = map(int, bbox)
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if match_found:
                matched_frames += 1

        display_cache(frame)
        return frame
    
    random.seed(42)
    person_colors = {}
    def get_color(person_id):
        if person_id not in person_colors:
            person_colors[person_id] = tuple(random.randint(0, 255) for _ in range(3))
        return person_colors[person_id]


    cap = cv2.VideoCapture(video_paths[0])
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    annotation_fps = 10
    video_fps = 59.94

    annotation_image_width = 1920
    annotation_image_height = 1080

    annotation_fps = 10  # GT is annotated at 10 FPS
    video_fps = 59.94
    view_num_target = 0
    annotation_image_width = 1920
    annotation_image_height = 1080
    num_annotation_frames = 400

    valid_ids = set(focus_id.values())

    for video_path in video_paths:
        if eval_mode:
            evaluation_mode(video_path=video_path,
                            annotation_image_width=annotation_image_width,
                            annotation_image_height=annotation_image_height,
                            valid_ids=valid_ids,
                            process_frame=process_frame,
                            get_color=get_color)
        else:
            presentation_mode(video_path=video_path,
                              duration=duration,
                              fps=fps,
                              total_frames=total_frames,
                              process_frame=process_frame,
                              out=out)
        cap.release()

    out.release()
    cv2.destroyAllWindows()
   
    print(f"Total Frames Processed: {total_frames}")
    print(f"Frames with Matches: {matched_frames}")
    mota_calc.calculate_mota()
lock_and_track_object(
    video_paths=[VIDEO_PATH1, VIDEO_PATH2, VIDEO_PATH3],
    model_path='../models/yolo11n.pt',
    output_path='output/locked_tracking_output.mp4',
    log_path='output/object_movement_log.json',
    eval_mode=False
)