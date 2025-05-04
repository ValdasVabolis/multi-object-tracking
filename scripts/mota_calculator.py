import os
import re
import json
from collections import defaultdict


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def normalize_video_name(video_name):
    match = re.match(r"(cam\d+)", video_name)
    return match.group(1) if match else video_name


def extract_view_num_from_filename(video_name):
    match = re.search(r"cam(\d+)", video_name)
    if match:
        return int(match.group(1)) - 1  # Adjust to 0-based index
    raise ValueError(f"Could not extract viewNum from {video_name}")


import json
import os
from collections import defaultdict

class MOTACalculator:
    def __init__(self, annotations_dir, iou_threshold=0.3):
        self.annotations_dir = annotations_dir
        self.iou_threshold = iou_threshold
        self.ground_truth = {}  # key: (video_name, frame_number)
        self.predictions = defaultdict(list)  # key: (video_name, frame_number)

        # STATIC: custom ID => ground truth personID
        self.person_mappings = {
            2: 9,
            4: 40,
            3: 8
        }

        # Reverse lookup: ground truth personID => custom ID
        self.person_id_to_custom = {v: k for k, v in self.person_mappings.items()}

    def log_prediction(self, video_name, frame_number, bbox, predicted_id):
        print("Predicted ID", predicted_id)
        """
        Logs a prediction if the predicted ID is in the static person_mappings.
        """
        if predicted_id not in self.person_mappings:
            return  # Ignore IDs not in the mapping
        print("Recorded person in frame {0} for ID {1} bbox={2}".format(frame_number, predicted_id, bbox))
        key = (normalize_video_name(video_name), frame_number)
        self.predictions[key].append({
            'id': predicted_id,
            'bbox': bbox
        })

    def ensure_ground_truth_loaded(self, video_name):
        normalized_name = normalize_video_name(video_name)
        if normalized_name in {key[0] for key in self.ground_truth.keys()}:
            return
        else:
            self.ground_truth = {}

        view_num = extract_view_num_from_filename(normalized_name)
        for frame_number in range(20):
            path = os.path.join(self.annotations_dir, f"{frame_number:08d}.json")
            if not os.path.exists(path):
                continue

            with open(path, 'r') as f:
                data = json.load(f)

            for person in data:
                person_id = person['personID']
                if person_id not in self.person_id_to_custom:
                    continue  # Skip IDs not in mapping

                for view in person['views']:
                    if view['viewNum'] == view_num and view['xmin'] != -1:
                        bbox = [view['xmin'], view['ymin'], view['xmax'], view['ymax']]
                        key = (normalized_name, frame_number)
                        if key not in self.ground_truth:
                            self.ground_truth[key] = []
                        self.ground_truth[key].append({
                            'id': person_id,
                            'bbox': bbox,
                            'matched': False
                        })
                        break

    def get_custom_id_for_person(self, person_id):
        return self.person_id_to_custom.get(person_id, None)

    def compute(self):
        TP, FP, FN, ID_switches, total_gt = 0, 0, 0, 0, 0
        print("Computing...", self.ground_truth.items())
        for key, gt_boxes in self.ground_truth.items():
            video_name, frame_number = key
            pred_boxes = self.predictions.get(key, [])
            for gt in gt_boxes:
                gt['matched'] = False

            total_gt += len(gt_boxes)
            matched_preds = set()

            for pred in pred_boxes:
                pred_id = pred['id']
                pred_bbox = pred['bbox']
                best_iou = 0
                best_gt = None
                for gt in gt_boxes:
                    if gt['matched']:
                        continue
                    iou = compute_iou(pred_bbox, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt
                print("Best GT was {0} with iou of {1} for pred {2}".format(best_gt, best_iou, pred))
                if best_iou >= self.iou_threshold and best_gt:
                    gt_id = best_gt['id']
                    best_gt['matched'] = True
                    expected_id = self.get_custom_id_for_person(gt_id)

                    # Ignore GT IDs not in person_mappings
                    if expected_id is None:
                        continue

                    if pred_id == expected_id:
                        TP += 1
                    else:
                        ID_switches += 1
                else:
                    print("ID {0} with bbox {1} was not found anywhere".format(pred_id, pred_bbox))
                    # Only count FP if predicted ID is known and mapped
                    if pred_id in self.person_mappings:
                        FP += 1
            FN += len([gt for gt in gt_boxes if not gt['matched']])

        mota = 1 - ((FN + FP + ID_switches) / total_gt) if total_gt else 0
        return {
            'MOTA': mota,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'ID_switches': ID_switches,
            'GT_total': total_gt,
            'person_mappings': self.person_mappings
        }