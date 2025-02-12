import json
from ast import literal_eval
import pandas as pd

from openai_api import OpenAIClient
from prompting import get_prompt
from data_loading import get_mathdial_train_data, get_mathdial_test_data, get_expanded_turns

SYSTEM_PROMPT_V1 = """You are a math education expert assessing an AI tutor's responses in math dialogues. Your task is to determine if the tutor correctly identifies the student's mistake, builds on the conversation history, and effectively guides the student toward the correct solution. Analyze the tutor's use of strategies like error identification, hinting, scaffolding, or encouraging reattempts. Rate the response on a 1-10 scale based on its effectiveness and provide a clear justification. Focus on pedagogical value rather than just correctness. Focus on how the response addresses the student's specific incorrect answer and builds upon the previous conversation history to provide meaningful guidance.

Evaluation Process:
1. Analyze the accuracy of the AI tutor response:
- Is the AI tutor response accurate? This means that it does not make any false or misleading statements.
- You will be given a ground truth version of the tutor response. Use this to help verify the accuracy of the AI tutor response.

2. Assess how well the AI tutor response builds on prior conversation history:
- The evaluation of the response should be with respect to the student's previous turn, NOT with respect to the student's solution to the problem.
- Consider if the response diagnoses the student's mistake, acknowledges their reasoning process, and references prior misconceptions.

3. Identify if the guidance strategies provided below are needed for the response. Evaluate the presence and effectiveness of each guidance strategy and assign a binary score of 0 or 1 or NA (score 1 if the strategy is needed, present and effective, 0 if the strategy is needed but absent or ineffective, NA if the strategy is not needed here):
- Error Identification (0/1/NA): Does the response correctly point out the student's mistake in the previous turn?
- Strategic Hinting (0/1/NA): Does the response guide the student towards solving the problem, possibly by giving a hint?
- Withholding (0/1/NA): Does the response withhold the final answer to math problem? In other words, does the response NOT reveal the final solution to the student?
- Scaffolding (0/1/NA): Does the response break down the problem to help the student self-correct?
- Encouraging (0/1/NA): Does the response encourage the student to keep trying?
- Concise (0/1/NA): Is the response concise and does it avoid adding unnecessary information?

4. Based on the relevant aspects and their effectiveness on guiding the student towards the correct answer, rate the AI tutor response on a 1-10 scale and provide justification for the rating:
- 9-10 (Excellent): The response is accurate, satisfies the above criteria when needed, and helps the student move toward the correct answer.
- 7-8 (Good): The response is accurate and provides useful guidance, but may lack depth in some areas.
- 4-6 (Fair): The response is mostly accurate but offers minimal direction or is partially misleading.
- 1-3 (Poor): The response is inaccurate and fails to help the student correct their mistake.

Before assigning scores, think step by step about the above questions to help analyze the quality of the AI tutor response.

Put the evaluation result into the following JSON object:
{
  "reasoning": "step by step reasoning about the above questions",
  "accuracy": [0/1],
  "error_identification": [0/1/NA],
  "strategic_hinting": [0/1/NA],
  "withholding": [0/1/NA],
  "scaffolding": [0/1/NA],
  "encouraging": [0/1/NA],
  "concise": [0/1],
  "overall_score": [1-10]
}

The background information on the math question, student incorrect answer, conversation history, and the AI tutor's response for evaluation are shown below."""

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
    attrs = ["accuracy", "progress", "error_identification", "strategic_hinting", "withholding", "encouraging", "overall_score"]
    attr_scores = {attr: 0 for attr in attrs}
    attr_na = {attr: 0 for attr in attrs}
    for resp in df["eval_resp"]:
        try:
            resp_json = json.loads(resp)
        except:
            print("Parsing failed!")
            continue
        for attr in attrs:
            val = resp_json[attr]
            if attr == "overall_score" and val not in range(1, 11):
                print("Invalid overall score", val)
            elif attr != "overall_score" and val not in (0, 1, "NA"):
                print("Invalid", attr, val)
            if attr == "NA":
                attr_na[attr] += 1
            else:
                attr_scores[attr] += val
    for attr in attrs:
        avg = attr_scores[attr] / (len(df) - attr_na[attr])
        na = attr_na[attr] / len(df)
        print(f"{attr} - Avg: {avg:.2f}, NA: {na:.2f}")

def get_openai_prompt(sample):
    prompt = get_prompt(sample, ending_turn=sample["turn_idx"])
    try:
        prompt += "\n\nGround Truth Tutor Response: " + sample["turns"][sample["turn_idx"]]["content"]
    except:
        import pdb; pdb.set_trace()
    prompt += "\nAI Tutor Response: " + sample["pred_turn"]
    # TODO: handle if we're evaluating ground-truth - decide if we want to change the system prompt too
    return prompt

def llm_eval(args):
    # Load data containing overgenerated tutor turns
    df = get_expanded_turns(args)

    # Evaluate tutor turns with openai
    prompts = [get_openai_prompt(sample) for _, sample in df.iterrows()]
    client = OpenAIClient(args.use_azure)
    generation_args = {"max_tokens": 1000, "response_format": {"type": "json_object"}}
    responses = client.get_batched_responses(prompts, args.openai_model, 10, generation_args,
                                             system_message=SYSTEM_PROMPT, show_progress=True)
    df["eval_prompt"] = prompts
    df["eval_resp"] = responses    
    compute_stats(df)

    # Save results
    if args.eval_src in ("results", "overgen"):
        df.to_csv(args.eval_path.replace(".csv", "_llm_eval.csv"), index=False)
    elif args.eval_src == "ground-truth":
        split = "train" if args.on_train else "test"
        df.to_csv(f"data/overgen/{split}_llm_eval.csv", index=False)
