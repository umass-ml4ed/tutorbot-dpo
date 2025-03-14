import pandas as pd

from llmkt_eval import llmkt_eval
from llm_eval import llm_eval

def eval_results(args):
    og_eval_src = args.eval_src
    og_eval_path = args.eval_path
    if args.use_cached_llmkt_eval:
        if og_eval_src == "ground-truth":
            split = "train" if args.on_train else "test"
            result_filename = f"results/{split}_llmkt_eval.csv"
        else:
            result_filename = og_eval_path.replace(".csv", "_llmkt_eval.csv")
        llmkt_df = pd.read_csv(result_filename)
        if args.truncate:
            llmkt_df = llmkt_df[:args.truncate]
        avg_score = llmkt_df["corr_pred"].mean()
        print(f"Avg correctness probability: {avg_score:.2f}")
    else:
        result_filename, avg_score = llmkt_eval(args)
    args.eval_src = "results"
    args.eval_path = result_filename
    _, stats_str = llm_eval(args)
    if og_eval_src == "ground-truth":
        split = "train" if args.on_train else "test"
        out_filename = f"results/metrics_{split}_ground-truth.txt"
    else:
        out_filename = og_eval_path.replace("outputs_", "metrics_").replace(".csv", ".txt")
    if args.use_cached_llmkt_eval and args.truncate:
        out_filename = out_filename.replace(".txt", f"_trunc{args.truncate}.txt")
    out_text = ""
    if args.truncate:
        out_text += f"Truncate: {args.truncate}\n"
    out_text += f"Avg correctness prediction: {avg_score:.4f}\nRubric stats:\n{stats_str}\n"
    with open(out_filename, "w") as f:
        f.write(out_text)
