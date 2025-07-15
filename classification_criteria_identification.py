import json
from openai import OpenAI
import argparse
from tqdm import tqdm

CRITERIA_PROMPT_TEMPLATE = """You are tasked with analyzing the reasoning strategies used in the following response. The response includes the thought process for solving a problem. Your goal is to extract and describe patterns based on various criteria that characterize the model's problem-solving strategy.
Please follow these guidelines:
1. Identify multiple *meaningful criteria* that differentiate reasoning strategies. Each criterion should have a clear and descriptive name that reflects a real aspect of the reasoning process. **Do not use generic placeholders like 'Criterion 1'.**
2. For each criterion, describe two contrasting *pattern types* (e.g., *Step-by-step* vs. *Outcome-first*, or *Concrete* vs. *Abstract*).
3. Present your analysis in the following format, using <patterns> and </patterns> tags to enclose the list:
<patterns>
Descriptive Criterion Name (Pattern A vs. Pattern B)
Descriptive Criterion Name (Pattern A vs. Pattern B)
...
Descriptive Criterion Name (Pattern A vs. Pattern B)
</patterns>
4. Do not include any explanations or commentary within the <patterns> tags.
5. The example format above is only a guide. You are encouraged to define your own diverse and insightful pattern criteria based on the given response.
Response: {response}"""

def main(dataset_path, output_path, openai_api_key, model="gpt-4o"):
    client = OpenAI(api_key=openai_api_key)
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    results = []
    for d in tqdm(dataset):
        response = d["response"]
        prompt = CRITERIA_PROMPT_TEMPLATE.format(response=response)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        d['criteria'] = completion.choices[0].message.content
        results.append(d)
    
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset .jsonl file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output .jsonl file")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    args = parser.parse_args()
    main(args.dataset_path, args.output_path, args.openai_api_key, args.model)

