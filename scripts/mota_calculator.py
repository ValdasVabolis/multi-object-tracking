import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple


class MOTACalculator:
    def __init__(self, annotation_dir: str, id_mapping: Dict[int, int], verbose: bool = True):
        self.annotation_dir = annotation_dir
        self.id_mapping = id_mapping  # internal_id -> external_id
        self.valid_external_ids = set(id_mapping.values())
        self.predictions = defaultdict(lambda: defaultdict(list))  # video_name -> frame_number -> list of predictions
        self.view_map = {f"cam{i+1}.mp4": i for i in range(7)}
        self.annotations_cache = {}
        self.verbose = verbose

    def log_prediction(self, video_name: str, frame_number: int, bbox: Tuple[int, int, int, int], internal_id: int):
        external_id = self.id_mapping.get(internal_id)
        if external_id is None:
            if self.verbose:
                print(f"[SKIP] Internal ID {internal_id} not mapped to external ID. Skipping prediction.")
            return

        self.predictions[video_name][frame_number].append({
            "bbox": bbox,
            "external_id": external_id
        })

    def _load_annotation(self, frame_number: int) -> List[Dict]:
        filename = f"{frame_number:08d}.json"
        filepath = os.path.join(self.annotation_dir, filename)
        if filepath in self.annotations_cache:
            return self.annotations_cache[filepath]

        if not os.path.exists(filepath):
            self.annotations_cache[filepath] = None
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)
            self.annotations_cache[filepath] = data
            return data

    def _iou(self, bbox1, bbox2) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def calculate_mota(self) -> float:
        total_gt = 0
        missed = 0
        false_positives = 0
        mismatches = 0

        if self.verbose:
            print("===== MOTA Calculation Started =====")

        for video_name, frames in self.predictions.items():
            view_num = self.view_map[video_name]
            if self.verbose:
                print(f"\n--- Evaluating {video_name} (view {view_num}) ---")

            for frame_number in sorted(frames.keys()):
                gt_data = self._load_annotation(frame_number)

                if gt_data is None:
                    # Skip frames without annotation file
                    if self.verbose:
                        print(f"Frame {frame_number:04d}: No GT annotation found. Skipping frame.")
                    continue

                preds = frames[frame_number]
                gt_bboxes = []
                gt_ids = []

                for obj in gt_data:
                    if obj['personID'] not in self.valid_external_ids:
                        continue
                    for view in obj['views']:
                        if view['viewNum'] == view_num and view['xmax'] != -1:
                            gt_bbox = (view['xmin'], view['ymin'], view['xmax'], view['ymax'])
                            gt_bboxes.append(gt_bbox)
                            gt_ids.append(obj['personID'])

                total_gt += len(gt_bboxes)
                matched_gt_indices = set()
                matched_pred_indices = set()

                frame_fp = 0
                frame_mm = 0
                frame_miss = 0
                frame_match = 0

                id_map = {2: 9}  # internal_id -> ground_truth_id
                target_gt_ids = set(id_map.values())  # {9}

                # Filter only relevant predictions and ground truth
                filtered_preds = [p for p in preds if p['external_id'] in id_map]
                filtered_gt_bboxes = []
                filtered_gt_ids = []
                for bbox, gid in zip(gt_bboxes, gt_ids):
                    if gid in target_gt_ids:
                        filtered_gt_bboxes.append(bbox)
                        filtered_gt_ids.append(gid)

                matched_gt_indices = set()
                matched_pred_indices = set()
                frame_match = frame_mm = frame_fp = 0

                for pred_idx, pred in enumerate(filtered_preds):
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt_bbox in enumerate(filtered_gt_bboxes):
                        if gt_idx in matched_gt_indices:
                            continue
                        iou = self._iou(pred['bbox'], gt_bbox)
                        if iou >= 0.1 and iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx

                    if best_gt_idx >= 0:
                        matched_gt_indices.add(best_gt_idx)
                        matched_pred_indices.add(pred_idx)
                        mapped_gt_id = id_map.get(pred['external_id'])

                        if mapped_gt_id != filtered_gt_ids[best_gt_idx]:
                            mismatches += 1
                            frame_mm += 1
                        else:
                            frame_match += 1
                    else:
                        false_positives += 1
                        frame_fp += 1

                frame_miss = len(filtered_gt_bboxes) - len(matched_gt_indices)
                missed += frame_miss

                if self.verbose:
                    print(f"Frame {frame_number:04d}: GT={len(filtered_gt_bboxes)} Match={frame_match} Miss={frame_miss} FP={frame_fp} Mismatch={frame_mm}")


        if total_gt == 0:
            if self.verbose:
                print("\n[INFO] No relevant ground truth objects found. Returning perfect MOTA=1.0")
            return 1.0

        mota = 1.0 - (missed + false_positives + mismatches) / total_gt
        if self.verbose:
            print("\n===== MOTA Calculation Complete =====")
            print(f"Total GT: {total_gt}, Missed: {missed}, FP: {false_positives}, Mismatches: {mismatches}")
            print(f"MOTA: {round(mota, 4)}")

        return round(mota, 4)