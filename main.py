from tracker import tracker_run
from evaluate import evaluate
from stantard_eval import run_standard_eval

def main():

    data_path = "dataset/MOT15/train/Venice-2"
    res_path = "results/results_Venice_2.txt"
    to_evaluate = True

    tracker_run(f"{data_path}/img1", res_path, display = True, export_result = to_evaluate)
    if to_evaluate:
        evaluate(f"{data_path}/gt/gt.txt", res_path)
        run_standard_eval(f"{data_path}/gt/gt.txt", res_path)

if __name__ == "__main__":
    main()