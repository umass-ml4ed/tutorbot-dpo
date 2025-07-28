# Tutorbot-DPO
This is the official code for the paper <a href="https://arxiv.org/abs/2503.06424">Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues</a>. We introduce a method for training tutorbots using DPO, where they are trained to maximize the likelihood of eliciting correct student responses while simultaneously adhering to pedagogical principles.

If you found our research or code useful for your work, then please cite us!
```
@InProceedings{scarlatos2025trainingllmbasedtutorsimprove,
title="Training LLM-Based Tutors to Improve Student Learning Outcomes in Dialogues",
author="Scarlatos, Alexander and Liu, Naiming and Lee, Jaewook and Baraniuk, Richard and Lan, Andrew",
editor="Cristea, Alexandra I. and Walker, Erin and Lu, Yu and Santos, Olga C. and Isotani, Seiji",
booktitle="Artificial Intelligence in Education",
year="2025",
publisher="Springer Nature Switzerland",
address="Cham",
pages="251--266",
isbn="978-3-031-98414-3"
}


```

## Setup

### Python
You can use pip to install the dependencies:
```
python -m venv tb
source tb/bin/activate
python -m pip install -r requirements.txt
```

### Dialogue KT
Our method relies on a dialogue-centered student model, LLMKT. To train LLMKT, clone https://github.com/umass-ml4ed/dialogue-kt and create a symlink to it in the top level of this repo:
```
ln -s <full path to repos>/dialogue-kt <full path to repos>/tutorbot-dpo/dialogue-kt
```

### Environment
Finally, set the following environment variables:
```
export PYTHONPATH=$PYTHONPATH:./dialogue-kt  # Make dialogue-kt code accessible to Python
export OPENAI_API_KEY=<your key here>        # For annotation and turn generation via OpenAI
export CUBLAS_WORKSPACE_CONFIG=:4096:8       # For enabling deterministic operations
```

## Train LLMKT Model
Both training and testing for our method rely on an LLMKT model, which you can train by running the following:
```
python -m dialogue_kt.main train --dataset mathdial --model_type lmkt --model_name lmkt_mathdial_r16_lr2e-4_mean-ar
```

## Create Training Set

The following will generate candidate tutor turns (from all 4 sources) and evaluate them using LLMKT and GPT-4o:
```
python main.py test_prompting --openai_model gpt-4o --detail_sys_prompt --on_train --truncate 4000  # GPT-4o
python main.py test --base_model meta-llama/Meta-Llama-3.1-8B-Instruct --on_train --truncate 4000   # Llama 3.1 8B
python main.py test --base_model meta-llama/Llama-3.2-3B-Instruct --on_train --truncate 4000        # Llama 3.2 3B
python main.py eval --eval_src ground-truth --on_train --truncate 4000                              # Human tutor turns
```

## Tutorbot Training and Testing

The following will train, test, and evaluate the different Llama-based methods used in the paper. Note that the distillation model is a prerequisite for the DPO model.

### SFT
```
python main.py sft --model_name sft-8b
```

### Distillation
```
python main.py sft --model_name distill-8b --sft_src results --sft_data_path results/outputs_train_baseline-4o_llmkt_eval_llm_eval.csv
```

### DPO
```
python main.py dpo --model_name dpo-8b --pt_model_name distill-8b --lr 3e-5 --epochs 1
```

### Baselines

Run the following for the untrained baselines used in the paper:
```
python main.py eval --eval_src ground-truth                                # Human tutor turns
python main.py test_prompting --openai_model gpt-4o --detail_sys_prompt    # GPT-4o
python main.py test --detail_sys_prompt                                    # Llama 3.1 8B
```
