# Tutorbot-DPO

## Training and Testing
### SFT
```
python main.py sft --model_name sft-8b
```

### DPO
```
python main.py dpo --model_name dpo-8b --pt_model_name sft-8b
```

### Test
```
python main.py test --model_name dpo-8b
```

# TODO: overgeneration, different evals
