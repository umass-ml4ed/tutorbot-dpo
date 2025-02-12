from transformers import PreTrainedTokenizer

SYSTEM_PROMPT = """You are a teacher helping a student solve a math problem. Your responses should be concise and conversational. Guide the student step-by-step, asking only one question at a time. Base your guidance on the student's incorrect solution and the conversation history."""

def get_prompt(sample, tokenizer: PreTrainedTokenizer = None, ending_turn: int = None):
    question = sample["question"]
    corr_solution = sample["ground_truth"]
    stud_solution = sample["student_incorrect_solution"].replace("\n\n", "\n")
    context = (
        f"Math problem: {question}\n"
        f"Correct solution: {corr_solution}\n"
        f"Student incorrect solution: {stud_solution}"
    )
    _ending_turn = ending_turn if ending_turn is not None else len(sample["turns"])
    if tokenizer is not None:
        starting_turn = 0
        if sample["turns"][0]["role"] == "user":
            starting_turn = 1
            context += f"\n\n{sample['turns'][0]['content']}"
        return tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
            *sample["turns"][starting_turn : _ending_turn]
        ], tokenize=False, add_generation_prompt=ending_turn is not None)
    else:
        prompt = context
        prompt += "\n\nConversation History:"
        for turn in sample["turns"][:_ending_turn]:
            prompt += "\nTutor: " if turn["role"] == "assistant" else "\nStudent: "
            prompt += turn["content"]
        return prompt
