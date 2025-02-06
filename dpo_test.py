import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import torch
from trl import DPOTrainer, DPOConfig
import random

def create_dataset(data):
    remove_label = True
    format_data = []

    format_conversation_student, format_conversation_teacher = [], []
    student_responses, teacher_responses = [], []
    for idx, row in data.iterrows():
        conversation = row["conversation"].split("|EOM|")
        format_conversation = []
        for sentence in conversation: 
            index = sentence.find(":")
            role = sentence[:index]
            response = sentence[index+1:]
            if role == "Teacher" and remove_label:
                response = " " + ")".join(response.split(")")[1:])
            format_conversation.append([role, response])
        format_data.append(format_conversation)
        conversation_student, response_student, conversation_teacher, response_teacher = [], [], [], []
        for idx, con in enumerate(format_conversation):
            if con[0] != "Teacher":
                previous = " \n".join([":".join(i) for i in format_conversation[:idx]])
                conversation_student.append(previous)
                response = format_conversation[idx][1].strip()
                response_student.append(response)
            if con[0] == "Teacher":
                previous = " \n".join([":".join(i) for i in format_conversation[:idx]])
                conversation_teacher.append(previous)
                response = format_conversation[idx][1].strip()
                response_teacher.append(response)
            
        format_conversation_student.append(conversation_student)
        student_responses.append(response_student)
        format_conversation_teacher.append(conversation_teacher)
        teacher_responses.append(response_teacher)

    data["format_data"] = format_data
    data["format_conversation_student"] = format_conversation_student
    data["student_responses"] = student_responses
    data["format_conversation_teacher"] = format_conversation_teacher
    data["teacher_responses"] = teacher_responses
    return data


def main(args):
    data = pd.read_csv("data/train.csv")
    responses = pd.read_csv(args.input_dir, index_col = 0)
    bad_responses = pd.read_csv(args.input_dir_bad, index_col=0)

    data = create_dataset(data)

    teacher_responses, question, profile, incorrect_answer, conversation = [], [], [], [], []
    for idx, row in data[0:args.input_count].iterrows():
        
        for idx, response in enumerate(row["teacher_responses"]):
            teacher_responses.append(response)
            question.append(row["question"])
            profile.append(row["student_profile"])
            incorrect_answer.append(row["student_incorrect_solution"])
            conversation.append(row["format_conversation_teacher"][idx])
    
    
    responses["real_responses"] = teacher_responses
    responses["profiles"] = profile
    responses["question"] = question
    responses["incorrect_answer"] = incorrect_answer
    responses["conversation"] = conversation
    
    bad_responses = bad_responses[bad_responses["number"] < args.input_count]
    bad_teacher_responses = []
    
    for idx, row in bad_responses.iterrows():
        bad_teacher_responses.append(row["responses"])
    
    responses["bad_responses"] = bad_teacher_responses

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
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
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
    tokenizer.padding_side = 'left'
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
        max_length=4096,
        max_prompt_length=4096,
        loss_type = "sigmoid",
        generate_during_eval=False,
        precompute_ref_log_probs=True,          # Avoid re-calculating reference model log probs every epoch
        output_dir=args.output_dir,             # directory to save
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
        push_to_hub=False,                      
        report_to="none"                        
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
    parser.add_argument("input_dir_bad", type=str, help="input data file (bad)")
    parser.add_argument("output_dir", type=str, help="output directory", default="test")
    parser.add_argument("input_count", type=int, help="number of data for input", default=500)
    parser.add_argument("epochs", type=int, help="number of epochs", default=1)
    args = parser.parse_args()

    main(args)
