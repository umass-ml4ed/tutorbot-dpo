from ast import literal_eval
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from dialogue_kt.training import get_lmkt_loss_packed, get_true_false_tokens
from dialogue_kt.models.lm import get_model
from dialogue_kt.kt_data_loading import LMKTCollatorPacked, get_dataloader
from dialogue_kt.prompting import kt_system_prompt, kt_user_prompt

from data_loading import get_expanded_turns, get_mathdial_test_data, get_mathdial_train_data

class DialogueKTArgs:
    def __init__(self):
        self.dataset = "mathdial"
        self.prompt_inc_labels = False
        self.agg = "mean-ar"

class LMKTDatasetPacked(Dataset):
    # Adapted from dialogue-kt code
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, args):
        self.data = []
        unique_idxs = set()
        for row_id, sample in data.iterrows():
            turn_pair_idx = sample["turn_idx"] // 2
            if turn_pair_idx >= len(sample["dialogue"]): # Skip when beyond annotated turns (happens when tutor turn ends dialogue)
                continue
            if "error" in sample["annotation"]: # Skip failed annotations
                continue
            # Get annotated KCs for this turn, skip if none since then can't make predictions
            cur_turn_annotation = sample["annotation"][f"turn {turn_pair_idx + 1}"]
            kcs = cur_turn_annotation["kcs"]
            if not kcs:
                continue
            # Swap tutor utterance at current turn for candidate utterance, copy so future turns still use ground truth
            dialogue = [*sample["dialogue"]]
            dialogue[turn_pair_idx] = {**dialogue[turn_pair_idx], "teacher": sample["pred_turn"]}
            # Create base prompt
            prompt = tokenizer.apply_chat_template([
                {"role": "system", "content": kt_system_prompt(args)},
                {"role": "user", "content": kt_user_prompt(sample, dialogue, turn_pair_idx + 1, None, args)},
            ], tokenize=False)
            # Add KC continuations to prompt
            kc_conts = [
                tokenizer.apply_chat_template([
                    {"role": "user", "content": kc},
                    {"role": "assistant", "content": f"\n"} # Newline would precede True or False prediction
                ], tokenize=False)
                for kc in kcs
            ]
            kc_conts = [" " + cont.split("user<|end_header_id|>\n\n")[1] for cont in kc_conts]
            prompt = prompt + "".join(kc_conts)
            unique_idxs.add(sample["index"])
            self.data.append({
                **sample,
                "row_id": row_id,
                "prompt": prompt,
                "label": cur_turn_annotation["correct"] or False,
                "kcs": kcs
            })
        print(f"Number of data points: {len(self.data)}, number of dialogues: {len(unique_idxs)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def llmkt_eval(args):
    # Load LLMKT model
    model, tokenizer = get_model("meta-llama/Meta-Llama-3.1-8B-Instruct", True, model_name="lmkt_mathdial_r16_lr2e-4_mean-ar", quantize=False)

    # Load data - get tutor turn candidates and merge with annotated dialogue KT data
    df = get_expanded_turns(args.eval_src, args.eval_path, args.truncate, args)
    src_split = "train" if args.eval_src == "overgen" or args.on_train or args.on_val else "test"
    data_cols = ["dialogue", "annotation", "meta_data"]
    annotated_df = pd.read_csv(f"dialogue-kt/data/annotated/mathdial_{src_split}_atc.csv", converters={col: literal_eval for col in data_cols})
    df = df.merge(annotated_df[["index", *data_cols]], on="index", how="inner")
    dialogue_kt_args = DialogueKTArgs()
    dataset = LMKTDatasetPacked(df, tokenizer, dialogue_kt_args)
    collator = LMKTCollatorPacked(tokenizer)
    data_loader = get_dataloader(dataset, collator, args.test_batch_size, False)

    # Evaluate tutor turn candidates with LLMKT model
    all_corr_probs = []
    true_token, false_token = get_true_false_tokens(tokenizer)
    for batch in tqdm(data_loader):
        batch["labels"] = batch["labels"].type(torch.bfloat16)
        with torch.no_grad():
            _, _, corr_probs = get_lmkt_loss_packed(model, batch, true_token, false_token, dialogue_kt_args)
        for sample, cp in zip(batch["meta_data"], corr_probs.tolist()):
            all_corr_probs.append({
                "row_id": sample["row_id"],
                "corr_pred": cp
            })

    # Compute stats and save results
    cp_df = pd.DataFrame(all_corr_probs)
    df = df.merge(cp_df, left_index=True, right_on="row_id", how="inner")
    avg_score = df["corr_pred"].mean()
    print(f"Avg correctness probability: {avg_score:.4f}")
    if args.eval_src in ("results", "overgen"):
        out_filename = args.eval_path.replace(".csv", "_llmkt_eval.csv")
    elif args.eval_src == "ground-truth":
        split = "train" if args.on_train else "test"
        out_filename = f"results/{split}_llmkt_eval.csv"
    df.to_csv(out_filename, index=False)
    return out_filename, avg_score
