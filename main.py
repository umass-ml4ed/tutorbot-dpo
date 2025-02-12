import argparse

from sft import sft
from dpo import dpo
from testing import test
from llm_eval import llm_eval
from llmkt_eval import llmkt_eval
from utils import initialize_seeds

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("mode", choices=["sft", "dpo", "test", "llm_eval", "llmkt_eval"])
    parser.add_argument("--truncate", type=int)
    parser.add_argument("--wandb", action="store_true")
    # Evaluation
    parser.add_argument("--openai_model")
    parser.add_argument("--use_azure", action="store_true")
    parser.add_argument("--eval_path")
    parser.add_argument("--eval_src", choices=["results", "overgen", "ground-truth"])
    # Modeling
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct") # "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training/Testing
    parser.add_argument("--on_train", action="store_true", help="Run testing/evaluation on train+val set; use for tutor turn overgeneration")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size at train-time")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Batch size at test-time")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--grad_accum_steps", default=32, type=int, help="Steps to accumulate gradients for")
    parser.add_argument("--r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_gen_tokens", type=int, default=1000, help="Maximum number of tokens to generate")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta parameter for DPO")
    parser.add_argument("--corr_weight", type=float, default=0.5, help="Amount to weigh correctness prediction in score calculation; between 0 and 1")

    args = parser.parse_args()
    if args.mode == "sft":
        assert args.model_name
        sft(args)
    elif args.mode == "dpo":
        assert args.model_name and args.pt_model_name
        dpo(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "llm_eval":
        assert args.openai_model and args.eval_src
        llm_eval(args)
    elif args.mode == "llmkt_eval":
        assert args.eval_src
        llmkt_eval(args)

if __name__ == "__main__":
    main()
