import os
import pandas as pd
import evaluate
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import get_base_model, get_model
from data_loading import get_mathdial_test_data, get_mathdial_train_data
from prompting import get_prompt
from eval_results import eval_results
from utils import device, get_model_file_suffix, run_gc

class TestingDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, args):
        self.data = []
        for index, sample in data.iterrows():
            starting_turn = 0 if sample["turns"][0]["role"] == "assistant" else 1
            for turn_idx in range(starting_turn, len(sample["turns"]), 2):
                self.data.append({
                    "index": index,
                    **sample,
                    "turn_idx": turn_idx,
                    "gt_turn": sample["turns"][turn_idx]["content"],
                    "prompt": get_prompt(sample, tokenizer, turn_idx, args)
                })
        print(f"Num dialogues: {len(data)}, num tutor turns: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class TestingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "left"
        tokens = self.tokenizer([sample["prompt"] for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "meta_data": batch
        }

def test(args):
    # Load model
    base_model, tokenizer = get_base_model(args.base_model, args.quantize)
    model = get_model(base_model, True, model_name=args.model_name, quantize=args.quantize)

    # Load data
    if args.on_train:
        train_data, val_data = get_mathdial_train_data()
        data = pd.concat([train_data, val_data])
    elif args.on_val:
        _, data = get_mathdial_train_data()
    else:
        data = get_mathdial_test_data()
    dataset = TestingDataset(data, tokenizer, args)
    if args.truncate:
        dataset.data = dataset.data[:args.truncate]
    collator = TestingCollator(tokenizer)
    data_loader = DataLoader(dataset, args.test_batch_size, collate_fn=collator, shuffle=False)

    # Generate tutor turns
    results = []
    for batch in tqdm(data_loader):
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_gen_tokens,
            do_sample=False
        )
        pred_turns = tokenizer.batch_decode(outputs[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
        results.extend([
            {**sample, "pred_turn": pred_turn} for sample, pred_turn in zip(batch["meta_data"], pred_turns)
        ])

    # Evaluate rouge
    rouge = evaluate.load("rouge")
    rouge_metrics = rouge.compute(
        predictions=[sample["pred_turn"] for sample in results],
        references=[sample["gt_turn"] for sample in results]
    )
    print("Turn similarity:", rouge_metrics)

    # Free up memory
    del base_model, model
    run_gc()

    # Save results and run different evals
    os.makedirs("results", exist_ok=True)
    results_df = pd.DataFrame(results)
    split = "train" if args.on_train else "val" if args.on_val else "test"
    out_filename = f"results/outputs_{split}_{get_model_file_suffix(args)}.csv"
    results_df.to_csv(out_filename, index=False)
    args.eval_src = "results"
    args.eval_path = out_filename
    eval_results(args)
