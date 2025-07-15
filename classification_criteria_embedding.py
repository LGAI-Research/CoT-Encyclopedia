import json
from openai import OpenAI
import argparse
from tqdm import tqdm

def main(input_path, output_path, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    with open(input_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    results = []
    for d in tqdm(dataset):
        criteria_text = d['criteria']

        embedding_response = client.embeddings.create(
            model="text-embedding-large",
            input=criteria_text
        )
        d["embedding"] = embedding_response.data[0].embedding
        results.append(d)

    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input .jsonl file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output .jsonl file")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.openai_api_key)