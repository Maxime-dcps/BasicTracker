import motmetrics as mm
import numpy as np
import pandas as pd

# Correction de compatibilité pour NumPy 2.0+
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda x: np.asarray(x, dtype=float)

def run_standard_eval(gt_file, res_file):
    # Charger les données
    gt_data = mm.io.loadtxt(gt_file, fmt="mot15-2D")
    res_data = mm.io.loadtxt(res_file, fmt="mot15-2D")

    acc = mm.MOTAccumulator(auto_id=True)

    # Utilisation de l'index de niveau 0 (Frame)
    frames = gt_data.index.get_level_values(0).unique()

    for f in frames:
        gt_frame = gt_data.loc[f]
        
        # Vérification si la frame existe
        if f in res_data.index.get_level_values(0):
            res_frame = res_data.loc[f]
        else:
            # Création d'un DataFrame vide avec les bonnes colonnes si pas de détection
            res_frame = pd.DataFrame(columns=['X', 'Y', 'Width', 'Height'])

        # Conversion explicite en float32
        gt_boxes = gt_frame[['X', 'Y', 'Width', 'Height']].values.astype(np.float32)
        res_boxes = res_frame[['X', 'Y', 'Width', 'Height']].values.astype(np.float32)

        # Calcul de la matrice de distance
        dist_matrix = mm.distances.iou_matrix(gt_boxes, res_boxes, max_iou=0.5)

        acc.update(
            gt_frame.index.get_level_values(0).values,
            res_frame.index.get_level_values(0).values,
            dist_matrix
        )

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'num_switches', 'idp', 'idr'], name='BasicTracker')
    print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))

if __name__ == "__main__":
    gt = "dataset/MOT15/train/PETS09-S2L1/gt/gt.txt"
    res = "results/results.txt" 
    run_standard_eval(gt, res)