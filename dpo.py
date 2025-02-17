from typing import List
from itertools import combinations
from ast import literal_eval
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig

from data_loading import get_mathdial_train_data
from model import get_base_model, get_model
from prompting import get_prompt
from llm_eval import RUBRIC_ATTRS
from testing import test
from utils import get_checkpoint_path

class DPODataset(Dataset):
    def __init__(self, data_srcs: List[pd.DataFrame], tokenizer: PreTrainedTokenizer, args):
        # Map each dialogue and turn index to all candidate turns and corresponding scores
        index_to_dialogue = {}
        for df in data_srcs:
            for _, sample in df.iterrows():
                dialogue = index_to_dialogue.setdefault(sample["index"], {
                    "sample": sample,
                    "turn_idx_to_cands": {}
                })
                turn_cands = dialogue["turn_idx_to_cands"].setdefault(sample["turn_idx"], {
                    "text": [],
                    "corr_scores": [],
                    "full_rubric": [],
                    "rubric_scores": [],
                    "scores": []
                })
                turn_cands["text"].append(sample["pred_turn"])
                corr_score = sample["corr_pred"]
                turn_cands["corr_scores"].append(corr_score)
                turn_cands["full_rubric"].append(sample["eval_resp"])
                if args.true_score:
                    rubric_score = np.mean([sample["eval_resp"][attr] for attr in RUBRIC_ATTRS[:-1]])
                else:
                    rubric_score = (sample["eval_resp"]["overall_score"] - 1) / 9
                turn_cands["rubric_scores"].append(rubric_score)
                if args.corr_weight is not None:
                    score = args.corr_weight * corr_score + (1 - args.corr_weight) * rubric_score
                    # if args.true_score:
                    #     score *= sample["eval_resp"]["accuracy"]
                    turn_cands["scores"].append(score)

        # Construct preference pairs
        self.data = []
        num_turns = 0
        for dialogue in index_to_dialogue.values():
            for turn_idx, turn_cands in dialogue["turn_idx_to_cands"].items():
                num_turns += 1
                prompt = get_prompt(dialogue["sample"], tokenizer, turn_idx, args)
                for idx0, idx1 in combinations(range(len(turn_cands["text"])), 2):
                    if args.corr_weight is not None:
                        # Compare weighted scores and take higher above threshold
                        diff = turn_cands["scores"][idx0] - turn_cands["scores"][idx1]
                        if diff > args.score_threshold:
                            chosen = turn_cands["text"][idx0]
                            rejected = turn_cands["text"][idx1]
                        elif diff < -args.score_threshold:
                            chosen = turn_cands["text"][idx1]
                            rejected = turn_cands["text"][idx0]
                        else:
                            continue
                    else:
                        # Compare individual scores and take strict domination above threshold
                        corr_diff = turn_cands["corr_scores"][idx0] - turn_cands["corr_scores"][idx1]
                        rubric_diff = turn_cands["rubric_scores"][idx0] - turn_cands["rubric_scores"][idx1]
                        if corr_diff > args.score_threshold and rubric_diff > args.score_threshold:
                            chosen = turn_cands["text"][idx0]
                            rejected = turn_cands["text"][idx1]
                        elif corr_diff < -args.score_threshold and rubric_diff < -args.score_threshold:
                            chosen = turn_cands["text"][idx1]
                            rejected = turn_cands["text"][idx0]
                        else:
                            continue
                    self.data.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected
                    })
        print(f"Num dialogues: {len(index_to_dialogue)}, num tutor turns: {num_turns}, num pref pairs: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def dpo(args):
    # Load model
    base_model, tokenizer = get_base_model(args.base_model, args.quantize)
    model = get_model(base_model, False, pt_model_name=args.pt_model_name, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize)
    if not args.pt_model_name:
        print("Using base model as reference model")

    # Load data
    data_src_map = {
        "gt": "results/train_llmkt_eval_llm_eval.csv", # Ground-truth
        "4o": "results/outputs_train_baseline-4o_llmkt_eval_llm_eval.csv", # GPT-4o
        "70b": "data/overgen/llama_70b_teacher_1_verified_llmkt_eval_llm_eval.csv", # Llama-70b pre-trained
        "8b": "results/outputs_train_meta-llama-Meta-Llama-3.1-8B-Instruct_llmkt_eval_llm_eval.csv", # Llama-8b pre-trained
        "3b": "results/outputs_train_meta-llama-Llama-3.2-3B-Instruct_llmkt_eval_llm_eval.csv", # Llama-3b pre-trained
    }
    data_src_files = [data_src_map[tag] for tag in args.dpo_data_srcs.split(",")]
    data_srcs = [pd.read_csv(filename, converters={"turns": literal_eval, "eval_resp": literal_eval}) for filename in data_src_files]
    train_data, val_data = get_mathdial_train_data()
    # Use indices to split overgenerated data into train/val
    train_data_srcs, val_data_srcs = [], []
    for split_data, split_list in [(train_data, train_data_srcs), (val_data, val_data_srcs)]:
        for df in data_srcs:
            df = df.merge(split_data[[]], left_on="index", right_index=True, how="inner")
            split_list.append(df)
    train_dataset = DPODataset(train_data_srcs, tokenizer, args)
    val_dataset = DPODataset(val_data_srcs, tokenizer, args)

    # Train
    config = DPOConfig(
        output_dir=get_checkpoint_path(args.model_name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        max_grad_norm=args.gc,
        warmup_ratio=0.1,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.test_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        report_to="wandb" if args.wandb else "none",
        # DPO-specific arguments
        beta=args.beta,
        loss_type=args.dpo_loss_type,
        rpo_alpha=args.rpo_alpha,
        use_weighting=args.use_wpo,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args.test_batch_size,
        model_adapter_name="default",
        ref_adapter_name="lora_ref" if args.pt_model_name else None,
        generate_during_eval=False
    )
    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=HFDataset.from_list(train_dataset.data),
        eval_dataset=HFDataset.from_list(val_dataset.data),
        processing_class=tokenizer
    )
    trainer.train()
    trainer.save_model()

    # Test
    test(args)
