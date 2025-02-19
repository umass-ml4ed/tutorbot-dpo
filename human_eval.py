import random
import argparse
from itertools import combinations
from ast import literal_eval
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau, rankdata, ttest_rel

NUM_DIALOGUES = 10
TURNS_PER_DIALOGUE = 5

def human_eval_create():
    rand = random.Random(221)
    srcs_files = [
        "results/test_llmkt_eval_llm_eval.csv",
        "results/outputs_test_baseline-4o_llmkt_eval_llm_eval.csv",
        "results/outputs_test_dpo-8b-allsrcs-cw.5-base4o-b.1_llmkt_eval_llm_eval.csv"
    ]
    dfs = [pd.read_csv(src_file, converters={"turns": literal_eval, "eval_resp": literal_eval}) for src_file in srcs_files]
    groups = dfs[0].groupby("index")
    candidate_idxs = [
        group.iloc[0]["index"] for _, group in groups
        if len(group) >= TURNS_PER_DIALOGUE + 1 and group.iloc[0]["qid"] != 6000047 and group.iloc[0]["index"] != 147
    ]
    # qid 6000047 is on an insensitive topic so skip it
    # index 147 dialogue starts with a keyboard mash from the tutor
    selected_idxs = rand.sample(candidate_idxs, NUM_DIALOGUES)
    out_data = []
    for index in selected_idxs:
        sample = dfs[0][dfs[0]["index"] == index].iloc[0]
        question = sample["question"]
        corr_solution = sample["ground_truth"]
        stud_solution = sample["student_incorrect_solution"].replace("\n\n", "\n")
        context = f"Math problem:\n{question}\n\nCorrect solution:\n{corr_solution}\n\nIncorrect student solution:\n{stud_solution}"
        for turn_idx in range(2, 2 * TURNS_PER_DIALOGUE + 2, 2): # First candidates on turn_idx = 2 (second tutor turn)
            cur_rows = [df[(df["index"] == index) & (df["turn_idx"] == turn_idx)].iloc[0] for df in dfs]
            method_order = [0, 1, 2]
            rand.shuffle(method_order)
            dialogue = ""
            for inner_turn_idx in range(turn_idx):
                turn = sample["turns"][inner_turn_idx]
                dialogue += ("Tutor: " if turn["role"] == "assistant" else "Student: ") + turn["content"] + "\n"
            dialogue += "\nNext Tutor Turn Candidates:"
            for m_idx, lett in zip(method_order, ["A", "B", "C"]):
                cand = cur_rows[m_idx]["pred_turn"].replace("\n", " ")
                dialogue += f"\n{lett}: {cand}"
            out_data.append({
                "context": context,
                "dialogue": dialogue,
                "method_order": method_order,
                "candidates": [row["pred_turn"] for row in cur_rows],
                "corr_preds": [row["corr_pred"] for row in cur_rows],
                "eval_resps": [row["eval_resp"] for row in cur_rows]
            })
    pd.DataFrame(out_data).to_csv("../human annotation/src_data.csv", index=False)

RANK_MAP = {
    "Most Likely": 1,
    "Second Most Likely": 2,
    "Least Likely": 3
}

def human_eval_analyze():
    # Collect GPT-4o scores and LLMKT ranks
    src_df = pd.read_csv("../human annotation/src_data.csv", converters={"eval_resps": literal_eval, "corr_preds": literal_eval, "method_order": literal_eval})
    ai_scores = []
    ai_corr_preds = []
    ai_ranks = []
    for _, sample in src_df.iterrows():
        ai_scores.extend([m_resp["overall_score"] for m_resp in sample["eval_resps"]])
        ai_corr_preds.append(sample["corr_preds"])
        ai_ranks.append(4 - rankdata(sample["corr_preds"]))

    # Collect scores and ranks from each annotator
    anno_df = pd.read_csv("../human annotation/TutorBot Annotation.csv")
    annotator_to_scores = [[] for _ in range(len(anno_df))]
    annotator_to_ranks = [[] for _ in range(len(anno_df))]
    annotator_to_method_to_scores = [[[] for _ in range(3)] for _ in range(len(anno_df))]
    annotator_to_method_to_ranks = [[[] for _ in range(3)] for _ in range(len(anno_df))]
    for anno_idx, row in anno_df.iterrows():
        all_scores = annotator_to_scores[anno_idx]
        all_ranks = annotator_to_ranks[anno_idx]
        for sample_idx, sample in src_df.iterrows():
            start_idx = 2 + sample_idx * 6
            scores = [row.iat[start_idx + m_idx] for m_idx in sample["method_order"]]
            ranks = [RANK_MAP[row.iat[start_idx + 3 + m_idx]] for m_idx in sample["method_order"]]
            all_scores.extend(scores)
            all_ranks.append(ranks)
            for m_idx in range(3):
                annotator_to_method_to_scores[anno_idx][m_idx].append(scores[m_idx])
                annotator_to_method_to_ranks[anno_idx][m_idx].append(ranks[m_idx])

    cutoff_to_valid_idxs = {
        cutoff: [
            idx for idx, corr_preds in enumerate(ai_corr_preds)
            if all([abs(cp0 - cp1) >= cutoff for cp0, cp1 in combinations(corr_preds, 2)])
        ]
        for cutoff in [0, 0.05, 0.1, 0.15]
    }

    # Compute annotator-annotator agreement
    for anno_idx_0, anno_idx_1 in combinations(range(len(anno_df)), 2):
        score_corr = pearsonr(annotator_to_scores[anno_idx_0], annotator_to_scores[anno_idx_1])
        print(f"Annotators {anno_idx_0} and {anno_idx_1}:")
        print(f"Score correlation: {score_corr}")
        anno_0_ranks = np.array(annotator_to_ranks[anno_idx_0])
        anno_1_ranks = np.array(annotator_to_ranks[anno_idx_1])
        for cutoff, valid_idxs in cutoff_to_valid_idxs.items():
            tau = kendalltau(anno_0_ranks[valid_idxs], anno_1_ranks[valid_idxs])
            print(f"Cutoff {cutoff} ({len(valid_idxs)} samples) - Rank Kendall's tau: {tau}")

    # Compute annotator-AI agreement
    for anno_idx in range(len(anno_df)):
        score_corr = pearsonr(annotator_to_scores[anno_idx], ai_scores)
        print(f"Annotator {anno_idx} and GPT-4o:")
        print(f"Score correlation: {score_corr}")
        anno_ranks = np.array(annotator_to_ranks[anno_idx])
        ai_ranks_np = np.array(ai_ranks)
        for cutoff, valid_idxs in cutoff_to_valid_idxs.items():
            tau = kendalltau(anno_ranks[valid_idxs], ai_ranks_np[valid_idxs])
            print(f"Cutoff {cutoff} ({len(valid_idxs)} samples) - Rank Kendall's tau: {tau}")

    # Compute per-method scores and ranks from the annotators
    for anno_idx in range(len(anno_df)):
        print(f"Annotator {anno_idx}:")
        method_to_scores = annotator_to_method_to_scores[anno_idx]
        method_to_ranks = annotator_to_method_to_ranks[anno_idx]
        for m_idx in range(3):
            print(f"Method {m_idx} avg. score: {np.mean(method_to_scores[m_idx])}, avg. rank: {np.mean(method_to_ranks[m_idx])}")
    print("Overall:")
    method_to_scores = [
        np.concatenate([annotator_to_method_to_scores[anno_idx][m_idx] for anno_idx in range(len(anno_df))])
        for m_idx in range(3)
    ]
    method_to_ranks = [
        np.concatenate([annotator_to_method_to_ranks[anno_idx][m_idx] for anno_idx in range(len(anno_df))])
        for m_idx in range(3)
    ]
    for m_idx in range(3):
        print(f"Method {m_idx} avg. score: {method_to_scores[m_idx].mean()}, avg. rank: {method_to_ranks[m_idx].mean()}")
    for m_idx0, m_idx1 in combinations(range(3), 2):
        scores_result = ttest_rel(method_to_scores[m_idx0], method_to_scores[m_idx1])
        ranks_result = ttest_rel(method_to_ranks[m_idx0], method_to_ranks[m_idx1])
        print(f"Methods {m_idx0} and {m_idx1} - Scores: {scores_result}, Ranks: {ranks_result}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "analyze"])
    args = parser.parse_args()

    if args.mode == "create":
        human_eval_create()
    elif args.mode == "analyze":
        human_eval_analyze()

if __name__ == "__main__":
    main()
