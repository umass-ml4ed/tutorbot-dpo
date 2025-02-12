from typing import Union, Tuple, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from utils import get_checkpoint_path

# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
)

def get_base_model(base_model_name: str, quantize: bool) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = "<|finetune_right_pad_id|>" # NOTE: this is a special padding token for llama, it's important to not set this to eot or any regularly occurring token or will mess with trl code
    print(f"Loading model {'with' if quantize else 'without'} quanitization: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        pad_token_id=tokenizer.pad_token_id,
        quantization_config=bnb_config if quantize else None,
        # f32 seems helpful for train/test time consistency when quantizing, bf16 performs best for non-quantized
        torch_dtype=torch.float32 if quantize else torch.bfloat16,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1
    return base_model, tokenizer

def get_model(model: PreTrainedModel, test: bool,
              model_name: Union[str, List[str]] = None, pt_model_name: str = None,
              r: int = None, lora_alpha: int = None,
              quantize: bool = False, use_gradient_checkpointing: bool = True) -> Union[PeftModel, PreTrainedModel]:
    if test and model_name:
        # Note we are loading adapter on quantized model and not merging
        # Recommended here - https://huggingface.co/docs/trl/main/en/dpo_trainer#downsides-to-merging-qlora-before-dpo-approach-2
        # Also prevents empty responses generated by Llama models
        model = PeftModel.from_pretrained(model, get_checkpoint_path(model_name))
    elif not test:
        if quantize:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)
        if pt_model_name:
            print(f"Initializing trainable model from pre-trained LoRA adapters: {pt_model_name}")
            model = PeftModel.from_pretrained(model, get_checkpoint_path(pt_model_name), is_trainable=True, adapter_name="default")
            model.load_adapter(get_checkpoint_path(pt_model_name), is_trainable=False, adapter_name="lora_ref")
        else:
            print("Initializing trainable model with new LoRA adapters")
            peft_config = LoraConfig(
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            model = get_peft_model(model, peft_config)
    else:
        print("Using inference-time model with pre-trained weights")
    return model
