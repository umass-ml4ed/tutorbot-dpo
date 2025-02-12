import re
from functools import partial
from ast import literal_eval
import pandas as pd

def process_dialogue(clean: bool, convo: str):
    turn_prefix_re = re.compile(r"^[a-zA-Z]+: (\([a-z]+\))?")
    outputs = []
    for turn in convo.split("|EOM|"):
        turn_text = turn_prefix_re.sub("", turn).strip()
        turn_dict = {
            "role": "assistant" if turn.startswith("Teacher") else "user",
            "content": turn_text
        }
        # Ensure no student (user) turns are repeated in the original data
        assert not outputs or turn_dict["role"] == "assistant" or outputs[-1]["role"] != turn_dict["role"]
        # Handle consecutive tutor turns
        if clean and outputs and turn_dict["role"] == outputs[-1]["role"]:
            # Sometimes turns are duplicated in the data - just skip these
            if turn_text == outputs[-1]["content"]:
                continue
            # Otherwise append text to previous turn
            if not outputs[-1]["content"].endswith((".", "!", "?")):
                outputs[-1]["content"] += "."
            outputs[-1]["content"] += " " + turn_text
        else:
            outputs.append(turn_dict)
        # Ensure that after cleaning there are no consecutive roles
        assert not clean or len(outputs) < 2 or outputs[-1]["role"] != outputs[-2]["role"]
    # Ensure all dialogues start with tutor (assistant) turns
    assert outputs[0]["role"] == "assistant"
    return outputs

def get_mathdial_train_data():
    train_df = pd.read_csv("data/src/mathdial/data/train.csv")
    train_df["turns"] = train_df["conversation"].apply(partial(process_dialogue, True))
    train_df["turns_all"] = train_df["conversation"].apply(partial(process_dialogue, False))
    train_df = train_df.sample(frac=1, random_state=221)
    return (
        train_df[:int(len(train_df) * .8)],
        train_df[int(len(train_df) * .8):]
    )

def get_mathdial_test_data():
    test_df = pd.read_csv("data/src/mathdial/data/test.csv")
    test_df["turns"] = test_df["conversation"].apply(partial(process_dialogue, True))
    test_df["turns_all"] = test_df["conversation"].apply(partial(process_dialogue, False))
    return test_df

def get_expanded_turns(args):
    # Load turn-level data from file based on source
    if args.eval_src == "results":
        df = pd.read_csv(args.eval_path, converters={"turns": literal_eval})
    elif args.eval_src == "overgen":
        df = pd.read_csv(args.eval_path, converters={"responses": literal_eval})
        df["pred_turn"] = df["responses"].apply(lambda x: x[1])
        train_df, val_df = get_mathdial_train_data()
        src_df = pd.concat([train_df, val_df])
        df = df.merge(src_df, how="left", left_on="number", right_index=True)
        indexed_data = []
        for _, seq in df.groupby("number"):
            turn_idx = 0
            all_turn_counter = 0
            all_turns = seq.iloc[0]["turns_all"]
            for _, row in seq.iterrows():
                indexed_data.append({
                    **row,
                    "index": row["number"],
                    "turn_idx": turn_idx
                })
                all_turn_counter += 1
                if all_turn_counter < len(all_turns) and all_turns[all_turn_counter]["role"] == "user":
                    all_turn_counter += 1
                    turn_idx += 2
        df = pd.DataFrame(indexed_data)
    elif args.eval_src == "ground-truth":
        if args.on_train:
            train_df, val_df = get_mathdial_train_data()
            df = pd.concat([train_df, val_df])
        else:
            df = get_mathdial_test_data()
        expanded_data = []
        for index, row in df.iterrows():
            for turn_idx, turn in enumerate(row["turns"]):
                if turn["role"] == "assistant":
                    expanded_data.append({
                        "index": index,
                        **row,
                        "pred_turn": turn["content"],
                        "turn_idx": turn_idx
                    })
        df = pd.DataFrame(expanded_data)
    if args.truncate:
        df = df[:args.truncate]
    return df
