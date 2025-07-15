import json
import argparse
import openai

RUBRIC_PROMPT_TEMPLATE = """Create a concise rubric for the following reasoning strategy criterion:
{criterion}
For each pattern, provide:
1. A clear, concise definition (2-3 sentences) that captures the essence of this reasoning strategy
2. 3-4 key characteristics that distinguish this pattern
3. 2 concrete examples of responses that demonstrate this pattern (keep examples brief, about 2-3 sentences each)
Focus on making the distinctions between patterns clear and easily identifiable. The definitions and examples should help evaluators quickly categorize model responses without ambiguity."""

def main(compressed_criteria_path, output_path, api_key, model="gpt-4o"):
    openai.api_key = api_key
    prompts = []
    with open(compressed_criteria_path, 'r') as f:
        for line in f:
            criterion = line.strip()
            if not criterion:
                continue
            rubric_prompt = RUBRIC_PROMPT_TEMPLATE.format(criterion=criterion)
            prompts.append(rubric_prompt)

    results = []
    for prompt in prompts:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800
            )
            results.append(response['choices'][0]['message']['content'])
        except Exception as e:
            print(f"OpenAI API error: {e}")
            results.append(f"[ERROR] {e}")

    with open(output_path, 'w') as f:
        for rubric in results:
            f.write(json.dumps(rubric) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rubric prompts from compressed criteria using OpenAI API.")
    parser.add_argument('--compressed_criteria_path', type=str, required=True, help='Path to txt file with compressed criteria.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated rubric prompts.')
    parser.add_argument('--api_key', type=str, required=True, help='Your OpenAI API key.')
    parser.add_argument('--model', type=str, default="gpt-4o", help='OpenAI model to use.')
    args = parser.parse_args()
    main(args.compressed_criteria_path, args.output_path, args.api_key, args.model)