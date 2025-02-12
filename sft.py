import pandas as pd
from transformers import PreTrainedTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset

from model import get_base_model, get_model
from data_loading import get_mathdial_train_data
from prompting import get_prompt
from testing import test
from utils import device, get_checkpoint_path

# For Llama 3, consts since no tokenizer variables point to these
BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007

class SFTDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.data = []
        excluded = 0
        for _, sample in data.iterrows():
            prompt = get_prompt(sample, tokenizer)
            if len(str(prompt)) < 6_000: # Exclude overly long dialogues to avoid OOM
                self.data.append({**sample.to_dict(), "prompt": prompt})
            else:
                excluded += 1
        print(f"Num dialogues: {len(self.data)} ({excluded} excluded)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SFTCollator:
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
    model = get_model(base_model, False, r=args.r, lora_alpha=args.lora_alpha, quantize=args.quantize)

    # Load data
    train_data, val_data = get_mathdial_train_data()
    train_dataset = SFTDataset(train_data, tokenizer)
    val_dataset = SFTDataset(val_data, tokenizer)
    collator = SFTCollator(tokenizer)

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
