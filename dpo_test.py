import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
from trl import DPOTrainer, DPOConfig
import random



def main(args):
    data = pd.read_csv("data/train.csv")
    responses = pd.read_csv(args.input_dir, index_col = 0)

    teacher_responses = []
    for idx, row in data[0:args.input_count].iterrows():
        for response in row["teacher_responses"]:
            teacher_responses.append(response)

    responses["real_responses"] = teacher_responses


    # Hugging Face model id
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" # replace with your model id
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dpo_data = []

    for idx, row in responses.iterrows():
        chosen_message = [{"content": row["real_responses"], "role": "assistant"}]
        rejected_message = [{"content": eval(row["responses"])[1], "role": "assistant"}]

        prompt = eval(row["prompts"])
        prompt[0]["content"] = "You are an math AI tutor respond to a student based on the conversation history to solve a math question."
        p = prompt[1]["content"][prompt[1]["content"].find('Math problem'): prompt[1]["content"].find('Student solution')] + prompt[1]["content"][prompt[1]["content"].find("Context:"): ]
        prompt[1]["content"] = p

        ### two versions
        dpo_data.append({
            "prompt": tokenizer.apply_chat_template(prompt, tokenize=False),
            "chosen": tokenizer.apply_chat_template(chosen_message, tokenize=False),
            "rejected": tokenizer.apply_chat_template(rejected_message, tokenize=False)
        })


    dpo_dataset = Dataset.from_list(dpo_data)
    dpo_dataset = dpo_dataset.train_test_split(test_size=0.1)

    train_dataset = dpo_dataset["train"]
    eval_dataset = dpo_dataset["test"]



    # Hugging Face model id
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct" # replace with your model id

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        use_cache=False, 
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation


    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=256,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM", 
    )


    args = DPOConfig(
        model_adapter_name="default",
        beta = 0.1,
        max_length=1024,
        max_prompt_length=4096,
        loss_type = "sigmoid",
        generate_during_eval=False,
        precompute_ref_log_probs=True,          # Avoid re-calculating reference model log probs every epoch
        output_dir=args.output_dir,             # directory to save and repository id (!!)
        num_train_epochs=args.epochs,           # number of training epochs
        per_device_train_batch_size=12,         # batch size per device during training
        per_device_eval_batch_size=4,           # batch size for evaluation
        gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        logging_steps=25,                       # log every 25 steps
        save_steps=500,                         # when to save checkpoint
        save_total_limit=2,                     # limit the total amount of checkpoints
        evaluation_strategy="steps",            # evaluate every 1000 steps
        eval_steps=50,                          # when to evaluate
        bf16=True,                              # use bfloat16 precision !!! VERY IMPORTANT
        push_to_hub=False,                      # push model to hub
        report_to="none"                        # present results (!!)
    )

    trainer = DPOTrainer(
        model,
        ref_model=None, # set to none since we use peft
        peft_config=peft_config,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # start training, the model will be automatically saved to the hub and the output directory
    trainer.train()
    
    # save model at the end of training
    trainer.save_model()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("intput_dir", type=str, help="input data file")
    parser.add_argument("output_dir", type=str, help="output directory", default="test")
    parser.add_argument("input_count", type=int, help="number of data for input", default=500)
    parser.add_argument("epochs", type=int, help="number of epochs", default=1)
    args = parser.parse_args()

    main(args)