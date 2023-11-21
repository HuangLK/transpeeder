import json
from tqdm import tqdm
import os
import argparse

GENERAL_PROMPT = "Add-Your-System-Prompt-Here"
UNIDIED_PROMPT = "<|prefix_begin|>{system_prompt}<|prefix_end|><|prompter|>{user_prompt}<|endoftext|><|assistant|>"
PREFIX_PROMPT = f"<|prefix_begin|>{GENERAL_PROMPT}<|prefix_end|>"

def read_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line.strip()))
    return data

def convert(input_path, output_path):
    '''
    Convert the data format from {"prompt":[p1,p2,p3...], "output":[o1,o2,o3...]} 
    to multi-round dialogue data format for transpeeder's dialog mode
    '''
    if os.path.exists(output_path):
        print("output dir exists!")
        return

    all_data = read_data(input_path)
    train = []
    for data in tqdm(all_data):
        prompt = data["prompt"]
        output = data["output"]
        q_prompt = PREFIX_PROMPT
        for idx in range(len(prompt) - 1):
            q_prompt += f"<|prompter|>{prompt[idx]}<|endoftext|><|assistant|>{output[idx]}<|endoftext|>"
        q_prompt += f"<|prompter|>{prompt[-1]}<|endoftext|><|assistant|>"
        out = {}
        out["prompt"] = q_prompt
        out["output"] = output[-1]
        train.append(out)

    with open(output_path, "w") as f:
        for line in train:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        help="Location of original data",
    )
    
    parser.add_argument(
        "--output_path",
        help="Location of data translated to the format of multi-round dialogue",
    )
    args = parser.parse_args()

    convert(
        input_path=args.input_path,
        output_path=args.output_path
    )

if __name__ == "__main__":
    main()