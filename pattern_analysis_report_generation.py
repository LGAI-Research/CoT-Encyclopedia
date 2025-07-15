import json
import argparse
from tqdm import tqdm
import openai

PROMPT_TEMPLATE = """You are an expert at analyzing reasoning strategies in model responses. You'll be provided with:
1. A rubric describing two distinct reasoning strategies (Pattern A and Pattern B)
2. A model response to analyze

Your task is to create a detailed analysis report that determines which pattern the response exhibits.

Analysis Process:
1. Carefully examine the response against both pattern definitions in the rubric
2. Identify specific elements, structures, and linguistic features in the response that align with either pattern
3. Note any mixed signals or elements that span both patterns
4. Determine which pattern (A or B) the response most closely matches

Report Structure:
1. **Initial Observations** (2-3 sentences summarizing key features of the reasoning approach)
2. **Evidence for Pattern A**:
- If applicable, quote 1-2 specific segments from the response that demonstrate Pattern A
- Explain how these segments match characteristics described in the rubric
3. **Evidence for Pattern B**:
- If applicable, quote 1-2 specific segments from the response that demonstrate Pattern B
- Explain how these segments match characteristics described in the rubric
4. **Pattern Determination**:
- Explain which pattern (A or B) is most dominant and why
- Address any aspects that show characteristics of both patterns
5. **Conclusion**:
- Clearly state the final pattern determination using the format: "Final pattern determination: [PATTERN NAME]"

Focus on concrete evidence from the response that matches specific elements from the rubric patterns.
Rubric: {rubric}
Response to analyze: {response}"""

def load_rubrics(rubric_path):
    rubrics = []
    with open(rubric_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # rubric_generation.py는 json.dumps로 저장하지만, 실제 내용은 string임
            try:
                rubric = json.loads(line)
                if isinstance(rubric, str):
                    rubrics.append(rubric)
                else:
                    rubrics.append(line)  # fallback
            except Exception:
                rubrics.append(line)
    return rubrics

def load_responses(dataset_path):
    responses = []
    with open(dataset_path, 'r') as f:
        for line in f:
            d = json.loads(line)
            # response 필드가 없으면 answer 등으로 fallback 가능
            response = d.get('response') or d.get('answer')
            if response:
                responses.append(response)
    return responses

def main(rubric_path, dataset_path, output_path, api_key, model="gpt-4o"):
    openai.api_key = api_key
    rubrics = load_rubrics(rubric_path)
    responses = load_responses(dataset_path)
    results = []
    total = len(rubrics) * len(responses)
    with tqdm(total=total) as pbar:
        for response in responses:
            for rubric in rubrics:
                prompt = PROMPT_TEMPLATE.format(rubric=rubric, response=response)
                try:
                    completion = openai.ChatCompletion.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=1000
                    )
                    report = completion['choices'][0]['message']['content']
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    report = f"[ERROR] {e}"
                results.append({"rubric": rubric, "response": response, "pattern_analysis_report": report})
                pbar.update(1)
                
    with open(output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pattern analysis reports for each rubric and response using OpenAI API.")
    parser.add_argument('--rubric_path', type=str, required=True, help='Path to the rubric file (jsonl or txt).')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset (jsonl).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the generated pattern analysis reports (jsonl).')
    parser.add_argument('--api_key', type=str, required=True, help='Your OpenAI API key.')
    parser.add_argument('--model', type=str, default="gpt-4o", help='OpenAI model to use.')
    args = parser.parse_args()
    main(args.rubric_path, args.dataset_path, args.output_path, args.api_key, args.model)