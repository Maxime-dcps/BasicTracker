from scipy.optimize import linear_sum_assignment
import numpy as np

def iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Intersection area
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    # Areas of both BBoxes
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Union = sum of areas - intersection
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area

def _match(tracks, detections, iou_threshold=0.3):
    num_tracks = len(tracks)
    num_dets = len(detections)
    
    if num_tracks == 0 or num_dets == 0:
        return [], list(range(num_tracks)), list(range(num_dets))
    
    # IoU matrix (we want to maximize, so cost = 1 - IoU to minimize)
    cost_matrix = np.zeros((num_tracks, num_dets))

    for i, t in enumerate(tracks):
        track_bbox = t.get_bbox()  # BBox predicted by Kalman
        for j, det_bbox in enumerate(detections):
            cost_matrix[i, j] = 1 - iou(track_bbox, det_bbox)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)  # Hungarian algorithm
    
    matches = []
    unmatched_tracks = list(range(num_tracks))
    unmatched_detections = list(range(num_dets))

    # Gating by minimum IoU
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= (1 - iou_threshold):  # IoU >= iou_threshold
            matches.append((r, c))
            unmatched_tracks.remove(r)
            unmatched_detections.remove(c)

    return matches, unmatched_tracks, unmatched_detections