import random
from ast import literal_eval
import pandas as pd

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

def human_eval_analyze():
    pass
