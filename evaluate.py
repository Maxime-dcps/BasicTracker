import numpy as np
import os
from matching import iou

def evaluate(gt_path, res_path, iou_threshold=0.5):
    # Load files
    gt_data = np.loadtxt(gt_path, delimiter=',')
    res_data = np.loadtxt(res_path, delimiter=',')

    frames = int(max(gt_data[:, 0]))
    
    total_fp = 0    # False Positives
    total_fn = 0    # False Negatives
    total_idsw = 0  # ID Switches
    total_gt_objects = 0
    
    # Dictionary to track which GT_ID is associated with which RES_ID
    # Format: {gt_id: last_res_id}
    current_associations = {}

    for f in range(1, frames + 1):
        # Extract data for current frame
        f_gt = gt_data[gt_data[:, 0] == f]
        f_res = res_data[res_data[:, 0] == f]
        
        total_gt_objects += len(f_gt)
        
        # Compute IoU matrix between GT and RES
        # Rows = GT, Columns = RES
        iou_matrix = np.zeros((len(f_gt), len(f_res)))
        for i, g in enumerate(f_gt):
            g_bbox = [g[2], g[3], g[2]+g[4], g[3]+g[5]]  # x1, y1, x2, y2
            for j, r in enumerate(f_res):
                r_bbox = [r[2], r[3], r[2]+r[4], r[3]+r[5]]
                iou_matrix[i, j] = iou(g_bbox, r_bbox)

        # For simplified version, take max IoU > threshold
        matches = []
        matched_gt = set()
        matched_res = set()

        # Sort by decreasing IoU for best associations first
        indices = np.where(iou_matrix >= iou_threshold)
        sorted_pairs = sorted(zip(indices[0], indices[1]), 
                             key=lambda x: iou_matrix[x[0], x[1]], reverse=True)

        for g_idx, r_idx in sorted_pairs:
            if g_idx not in matched_gt and r_idx not in matched_res:
                matched_gt.add(g_idx)
                matched_res.add(r_idx)
                
                gt_id = int(f_gt[g_idx][1])
                res_id = int(f_res[r_idx][1])
                
                # --- CHECK ID SWITCH ---
                if gt_id in current_associations:
                    if current_associations[gt_id] != res_id:
                        total_idsw += 1
                
                # Update current association
                current_associations[gt_id] = res_id

        # Count errors
        total_fp += (len(f_res) - len(matched_res))
        total_fn += (len(f_gt) - len(matched_gt))

    # Final MOTA calculation
    mota = 1 - (total_fn + total_fp + total_idsw) / total_gt_objects
    
    print(f"--- Evaluation Results ---")
    print(f"MOTA: {mota:.2%}")
    print(f"False Negatives: {total_fn}")
    print(f"False Positives: {total_fp}")
    print(f"ID Switches: {total_idsw}")
    print(f"Total GT Objects: {total_gt_objects}")


if __name__ == "__main__":
    evaluate("dataset/MOT15/train/PETS09-S2L1/gt/gt.txt", "results/results.txt")