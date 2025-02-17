import json
import pandas as pd
import numpy as np

from openai_api import OpenAIClient
from prompting import get_prompt
from data_loading import get_expanded_turns

RUBRIC_ATTRS = ["accuracy", "progress", "error_identification", "strategic_hinting", "withholding", "encouraging", "overall_score"]

SYSTEM_PROMPT = """You are a math education expert assessing an AI tutor's responses in math dialogues. Your task is to determine if the tutor correctly identifies the student's mistake, builds on the conversation history, and effectively guides the student toward the correct solution. Focus on pedagogical value rather than just correctness.

Evaluation Process:
1. Analyze the *accuracy* of the AI tutor response:
- Is the AI tutor response accurate? This means that it does not make any false or misleading statements.
- You will be given a ground truth version of the tutor response. Use this to help verify the accuracy of the AI tutor response.

2. Assess how well the AI tutor helps the student *progress* towards the correct answer:
- Does the AI tutor response help the student make progress in some way? This could be by giving a hint, addressing a misconception, or asking the student to clarify.
- The response should be novel in some way. It should not simply reiterate concepts that the student already understands, and should not repeat previous failed strategies for helping the student.
- Analyze the AI tutor response in the context of the entire dialogue history.

3. Analyze the use of the guidance strategies below. Evaluate the presence and effectiveness of each guidance strategy and assign a binary score of 0 or 1 (score 1 if the strategy is present and effective and 0 if the strategy is absent or ineffective):
- Error Identification (0/1): Does the response correctly point out the student's mistake in the previous turn?
- Strategic Hinting (0/1): Does the response give the student some new information, such as a hint, to help them progress?
- Withholding (0/1): Does the response withhold the final answer to math problem? In other words, does the response NOT reveal the final solution to the student?
- Encouraging (0/1): Does the response encourage the student to keep trying?

4. Based on the relevant aspects and their effectiveness on guiding the student towards the correct answer, rate the AI tutor response on a 1-10 scale and provide justification for the rating:
- 9-10 (Excellent): The response is accurate, satisfies the above criteria, and helps the student move toward the correct answer.
- 7-8 (Good): The response is accurate and provides useful guidance, but may lack depth in some areas.
- 4-6 (Fair): The response is mostly accurate but offers minimal direction or is partially misleading.
- 1-3 (Poor): The response is inaccurate and fails to help the student correct their mistake.

Before assigning scores, think step by step about the above questions to help analyze the quality of the AI tutor response.

Put the evaluation result into the following JSON object:
{
  "reasoning": "step by step reasoning about the above questions",
  "accuracy": [0/1],
  "progress": [0/1],
  "error_identification": [0/1],
  "strategic_hinting": [0/1],
  "withholding": [0/1],
  "encouraging": [0/1],
  "overall_score": [1-10]
}

The background information on the math question, student incorrect answer, conversation history, and the AI tutor's response for evaluation are shown below."""

def compute_stats(df: pd.DataFrame):
    attr_scores = {attr: 0 for attr in RUBRIC_ATTRS}
    attr_scores["true_score"] = 0
    stats = []
    for resp in df["eval_resp"]:
        try:
            resp_json = json.loads(resp)
        except:
            print("Parsing failed!")
            continue
        for attr in RUBRIC_ATTRS:
            val = resp_json[attr]
            if attr == "overall_score" and val not in range(1, 11):
                print("Invalid overall score", val)
            elif attr != "overall_score" and val not in (0, 1):
                print("Invalid", attr, val)
            attr_scores[attr] += val
        # attr_scores["true_score"] += resp_json["accuracy"] * np.mean([resp_json[attr] for attr in RUBRIC_ATTRS[:-1]])
        attr_scores["true_score"] += np.mean([resp_json[attr] for attr in RUBRIC_ATTRS[:-1]])
    for attr in RUBRIC_ATTRS + ["true_score"]:
        avg = attr_scores[attr] / len(df)
        stats.append(f"{attr} - Avg: {avg:.4f}")
    stats_str = "\n".join(stats)
    print(stats_str)
    return stats_str

def get_openai_prompt(sample):
    prompt = get_prompt(sample, ending_turn=sample["turn_idx"])
    prompt += "\n\nGround Truth Tutor Response: " + sample["turns"][sample["turn_idx"]]["content"]
    prompt += "\nAI Tutor Response: " + sample["pred_turn"]
    return prompt

def llm_eval(args):
    # Load data containing overgenerated tutor turns
    df = get_expanded_turns(args.eval_src, args.eval_path, args.truncate, args)

    # Evaluate tutor turns with openai
    prompts = [get_openai_prompt(sample) for _, sample in df.iterrows()]
    client = OpenAIClient(args.use_azure)
    generation_args = {"max_tokens": 1000, "response_format": {"type": "json_object"}}
    responses = client.get_batched_responses(prompts, args.openai_model, 10, generation_args,
                                             system_message=SYSTEM_PROMPT, show_progress=True)
    df["eval_prompt"] = prompts
    df["eval_resp"] = responses    
    stats_str = compute_stats(df)

    # Save results
    if args.eval_src in ("results", "overgen"):
        out_filename = args.eval_path.replace(".csv", "_llm_eval.csv")
    elif args.eval_src == "ground-truth":
        split = "train" if args.on_train else "test"
        out_filename = f"results/{split}_llm_eval.csv"
    df.to_csv(out_filename, index=False)
    return out_filename, stats_str
