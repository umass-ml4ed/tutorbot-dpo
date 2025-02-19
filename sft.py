import pandas as pd
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

from model import get_base_model, get_model
from data_loading import get_mathdial_train_data, get_expanded_turns
from prompting import get_prompt
from testing import test
from utils import device, get_checkpoint_path

# For Llama 3, consts since no tokenizer variables point to these
BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007

MAX_LEN = 6_000 # Exclude overly long prompts to avoid OOM

class SFTCombinedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, args):
        self.data = []
        excluded = 0
        num_turns = 0
        for _, sample in data.iterrows():
            num_turns += round(len(sample["turns"]) / 2)
            prompt = get_prompt(sample, tokenizer, args=args)
            if len(prompt) < MAX_LEN:
                self.data.append({**sample, "prompt": prompt})
            else:
                excluded += 1
        print(f"Num dialogues: {len(self.data)} ({excluded} excluded), num turns: {num_turns}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SFTCombinedCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer([sample["prompt"] for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        # Create labels - mask out all but assistant turns
        labels = tokens.input_ids.clone()
        labels[tokens.attention_mask == 0] = -100 # Mask padding region
        for idx in range(len(labels)):
            boh_idxs = (labels[idx] == BOH_TOKEN_ID).nonzero()
            eoh_idxs = (labels[idx] == EOH_TOKEN_ID).nonzero()
            labels[idx, :eoh_idxs[2] + 1] = -100 # Mask labels up to end of first assistant header
            for header_ct in range(3, len(boh_idxs), 2):
                # Mask labels between start of each user header to end of subsequent assistant header
                end_idx = eoh_idxs[header_ct + 1] if header_ct + 1 < len(eoh_idxs) else labels.shape[1] # In case dialogue ends with a user turn
                labels[idx, boh_idxs[header_ct] : end_idx + 1] = -100
        return {
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": labels
        }

class SFTExpandedDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer, args):
        self.data = []
        excluded = 0
        for _, sample in data.iterrows():
            prompt = get_prompt(sample, tokenizer, sample["turn_idx"], args)
            if len(prompt + sample["pred_turn"]) < MAX_LEN:
                self.data.append({**sample, "prompt": prompt, "label": sample["pred_turn"] + tokenizer.eos_token})
            else:
                excluded += 1
        print(f"Num turns: {len(self.data)} ({excluded} excluded)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SFTExpandedCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(
            [sample["prompt"] + sample["label"] for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = tokens.input_ids
        attn_mask = tokens.attention_mask
        prompt_tokens = self.tokenizer(
            [sample["prompt"] for sample in batch],
            return_tensors="pt", padding=True, add_special_tokens=False)
        prompt_lens = prompt_tokens.attention_mask.sum(dim=1)
        labels = input_ids.clone()
        labels[attn_mask == 0] = -100
        label_mask = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1) < prompt_lens.unsqueeze(1)
        labels[label_mask] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels
        }

def get_training_args(args):
    return TrainingArguments(
        output_dir=get_checkpoint_path(args.model_name),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        max_grad_norm=args.gc or None,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        per_device_eval_batch_size=args.test_batch_size,
        eval_accumulation_steps=4,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb" if args.wandb else "none"
    )

def sft(args):
    # Load model
    base_model, tokenizer = get_base_model(args.base_model, args.quantize)
    model = get_model(base_model, False, pt_model_name=args.pt_model_name, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize)

    # Load data
    train_data, val_data = get_mathdial_train_data()
    if args.sft_src == "ground-truth":
        train_dataset = SFTCombinedDataset(train_data, tokenizer, args)
        val_dataset = SFTCombinedDataset(val_data, tokenizer, args)
        collator = SFTCombinedCollator(tokenizer)
    else:
        data = get_expanded_turns(args.sft_src, args.sft_data_path, None, args)
        train_data = data.merge(train_data[[]], left_on="index", right_index=True, how="inner")
        val_data = data.merge(val_data[[]], left_on="index", right_index=True, how="inner")
        train_dataset = SFTExpandedDataset(train_data, tokenizer, args)
        val_dataset = SFTExpandedDataset(val_data, tokenizer, args)
        collator = SFTExpandedCollator(tokenizer)

    # Train
    trainer = Trainer(
        model=model,
        args=get_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator
    )
    trainer.train()
    trainer.save_model()

    # Test
    test(args)
