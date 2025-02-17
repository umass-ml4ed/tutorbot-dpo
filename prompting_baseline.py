from data_loading import get_expanded_turns
from prompting import get_prompt, DETAILED_SYSTEM_PROMPT
from openai_api import OpenAIClient
from eval_results import eval_results

def clean_resp(resp: str):
    return resp

def prompting_baseline(args):
    data = get_expanded_turns("ground-truth", None, args.truncate, args)
    prompts = [get_prompt(sample, ending_turn=sample["turn_idx"]) for _, sample in data.iterrows()]
    client = OpenAIClient(args.use_azure)
    generation_args = {"max_tokens": 300}
    responses = client.get_batched_responses(prompts, args.openai_model, 10, generation_args,
                                             system_message=DETAILED_SYSTEM_PROMPT, show_progress=True)
    responses = [clean_resp(resp) for resp in responses]
    data["pred_turn"] = responses
    split = "train" if args.on_train else "test"
    out_filename = f"results/outputs_{split}_baseline-{args.openai_model}.csv"
    data.to_csv(out_filename, index=False)
    args.eval_src = "results"
    args.eval_path = out_filename
    eval_results(args)
