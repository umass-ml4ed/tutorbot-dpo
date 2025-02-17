# Tutorbot-DPO

## Setup

### For 70B
Special setup needed to run 70B model with unsloth.
https://docs.unsloth.ai/get-started/installing-+-updating/pip-install
```
python -m venv tbu
source tbu/bin/activate
python -m pip install -r unsloth_requirements.txt
python -m pip install "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

## Overgeneration and Evaluation

### Overgenerate turns

### LLMKT Evaluation

### Rubric Evaluation

## Training and Testing
### SFT
```
python main.py sft --model_name sft-8b
```

### DPO
```
python main.py dpo --model_name dpo-8b --pt_model_name sft-8b --lr 3e-5 --epochs 3
```

### Test
```
python main.py test --model_name dpo-8b
```

### Evaluate Results
