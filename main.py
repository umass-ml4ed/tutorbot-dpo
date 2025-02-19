import argparse

from sft import sft
from dpo import dpo
from testing import test
from prompting_baseline import prompting_baseline
from llm_eval import llm_eval
from llmkt_eval import llmkt_eval
from eval_results import eval_results
from utils import initialize_seeds, bool_type

DEFAULTS_3B = {
    "train_batch_size": 2,
    "grad_accum_steps": 32,
    "test_batch_size": 16
}

DEFAULTS_8B = {
    "train_batch_size": 1,
    "grad_accum_steps": 64,
    "test_batch_size": 8
}

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # General
    parser.add_argument("mode", choices=["sft", "dpo", "test", "test_prompting", "eval", "llm_eval", "llmkt_eval"])
    parser.add_argument("--truncate", type=int)
    parser.add_argument("--wandb", action="store_true")
    # Evaluation
    parser.add_argument("--openai_model", default="4o")
    parser.add_argument("--use_azure", type=bool_type, default=True)
    parser.add_argument("--eval_src", choices=["results", "overgen", "ground-truth"])
    parser.add_argument("--eval_path")
    parser.add_argument("--use_cached_llmkt_eval", action="store_true")
    # Modeling
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct") # "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--detail_sys_prompt", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    # Training/Testing
    parser.add_argument("--on_train", action="store_true", help="Run testing/evaluation on train+val set; use for tutor turn overgeneration")
    parser.add_argument("--on_val", action="store_true", help="Run testing/evaluation on val set; use for hyperparameter search")
    parser.add_argument("--train_batch_size", type=int, help="Batch size at train-time")
    parser.add_argument("--test_batch_size", type=int, help="Batch size at test-time")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients for")
    parser.add_argument("--r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_gen_tokens", type=int, default=300, help="Maximum number of tokens to generate")
    parser.add_argument("--sft_src", choices=["results", "overgen", "ground-truth"], default="ground-truth")
    parser.add_argument("--sft_data_path")
    # DPO
    parser.add_argument("--dpo_data_srcs", default="gt,4o,8b,3b")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter for DPO")
    parser.add_argument("--corr_weight", type=float, default=0.5, help="Amount to weigh correctness prediction in score calculation; between 0 and 1")
    parser.add_argument("--score_threshold", type=float, default=0.1, help="Score threshold to surpass to form a preference pair")
    parser.add_argument("--true_score", type=bool_type, default=False)
    parser.add_argument("--dpo_loss_type", default="sigmoid")
    parser.add_argument("--rpo_alpha", type=float)
    parser.add_argument("--use_wpo", type=bool_type, default=False)

    args = parser.parse_args()

    defaults = args.base_model and (DEFAULTS_3B if "-3B-" in args.base_model else DEFAULTS_8B)
    if defaults:
        for k, v in defaults.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    if args.mode == "sft":
        assert args.model_name
        sft(args)
    elif args.mode == "dpo":
        assert args.model_name
        dpo(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "test_prompting":
        assert args.openai_model
        prompting_baseline(args)
    elif args.mode == "llm_eval":
        assert args.openai_model and args.eval_src
        llm_eval(args)
    elif args.mode == "llmkt_eval":
        assert args.eval_src
        llmkt_eval(args)
    elif args.mode == "eval":
        assert args.openai_model and args.eval_src
        eval_results(args)

if __name__ == "__main__":
    main()
