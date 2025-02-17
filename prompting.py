from transformers import PreTrainedTokenizer

BASIC_SYSTEM_PROMPT = """You are a teacher helping a student solve a math problem. Your responses should be concise and conversational. Guide the student step-by-step, asking only one question at a time. Base your guidance on the student's incorrect solution and the conversation history."""

DETAILED_SYSTEM_PROMPT = """You are a math education expert, and your job is to tutor a student to help them solve a math problem. You will be given the math problem, the student's incorrect answer to that problem, and the conversation so far. Please write a single, concise statement to the student to guide them towards solving the problem. Your response should meet these following criteria:
- Ensure Correctness: It should be highly likely that the student will be able to correctly answer your response.
- Accurate: You should be factually accurate and not contain any misleading statements.
- Encouraging Progress: You should help the student make progress on the problem. This could be by giving a hint, addressing a misconception, or asking the student to clarify. Your response should be novel in some way. It should not simply reiterate concepts that the student already understands, and should not repeat previous failed strategies for helping the student.
- Error Identification: If the student made an error in their previous turn, you should identify the error or the misconcepton underlying it.
- Strategic Hinting: You should provide some guidance, such as a hint, to help the student progress.
- Withholding: You should NOT reveal the final solution to the student.
- Encouraging: You should encourage the student to keep trying."""

def get_prompt(sample, tokenizer: PreTrainedTokenizer = None, ending_turn: int = None, args = None):
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
        system_prompt = DETAILED_SYSTEM_PROMPT if args.detail_sys_prompt else BASIC_SYSTEM_PROMPT
        return tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
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
