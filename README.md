# Tutorbot-DPO

## Setup

### Dialogue KT
Clone repo and create symlink and set PYTHONPATH

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
