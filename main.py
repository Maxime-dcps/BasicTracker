from tracker import tracker_run
from evaluate import evaluate
from stantard_eval import run_standard_eval

def main():

    MOT15_path = [
        "ADL-Rundle-6", 
        "ADL-Rundle-8", 
        "ETH-Bahnhof", 
        "ETH-Pedcross2", 
        "ETH-Sunnyday", 
        "KITTI-13", 
        "KITTI-17", 
        "PETS09-S2L1", 
        "TUD-Campus", 
        "TUD-Stadtmitte", 
        "Venice-2"
    ]
    data_path = "dataset/MOT15/train"
    to_evaluate = True

    # Initialisation des compteurs globaux
    global_fn = 0
    global_fp = 0
    global_idsw = 0
    global_gt_objects = 0

    for sequence in MOT15_path:
        img_dir = f"{data_path}/{sequence}/img1"
        res_file = f"results/results_{sequence}.txt"
        gt_file = f"{data_path}/{sequence}/gt/gt.txt"

        print(f"Tracking sequence: {sequence}...")
        tracker_run(img_dir, res_file, display=True, export_result=to_evaluate)

        # Évaluer la séquence
        if to_evaluate:
            # On récupère les scores de la séquence
            mota, fn, fp, idsw, gt_obj = evaluate(gt_file, res_file)
            
            # On accumule pour le score global
            global_fn += fn
            global_fp += fp
            global_idsw += idsw
            global_gt_objects += gt_obj
            
            run_standard_eval(gt_file, res_file)

    if to_evaluate and global_gt_objects > 0:
        final_mota = 1 - (global_fn + global_fp + global_idsw) / global_gt_objects

        print(f"\n--- TOTAL MOT15 TRAIN RESULTS ---")
        print(f"Overall MOTA: {final_mota:.2%}")
        print(f"Total False Negatives: {global_fn}")
        print(f"Total False Positives: {global_fp}")
        print(f"Total ID Switches: {global_idsw}")
        print(f"Total GT Objects: {global_gt_objects}")

if __name__ == "__main__":
    main()