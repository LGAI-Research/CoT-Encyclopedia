# CoT-Encyclopedia
This is the official github repository for The CoT Encyclopedia: Analyzing, Predicting, and Controlling how a Reasoning Model will Think. <br><br>
Traditional methods use fixed criteria to identify strategies but offer limited guidance for improving reasoning. The CoT Encyclopedia takes a bottom-up approach, uncovering diverse, task-specific strategies and enabling flexible analysis and actionable insights to enhance model performance.

## Overview
![cot_encyclopedia_overview](overview.png)
The framework constructs a taxonomy of reasoning strategies through five key stages: (1) Classification Criteria Identification – diverse reasoning criteria are identified from model-generated CoTs; (2) Classification Criteria Embedding – these criteria are converted into semantic embeddings; (3) Criteria Compression via Hierarchical Clustering – semantically similar criteria are clustered to form distinct representative categories; (4) Rubric Generation – contrastive rubrics are created to describe and distinguish opposing reasoning patterns within each criterion; (5) Analysis Report Generation – model responses are classified using the rubrics, producing comprehensive reports that interpret their reasoning behaviors. The framework also supports practical use cases such as reasoning pattern analysis and optimal strategy control for performance improvement.

## Setup
```bash
git clone https://github.com/sylee0520/CoT-Encyclopedia.git
conda create -n cot_encyclopedia python=3.10
conda activate cot_encyclopedia
pip install openai
cd CoT-Encyclopedia
```

## How to Run the CoT-Encyclopedia
Please configure the environment variables before execution. For example:
```bash
DATASET_PATH="dummy_dataset.jsonl"
RAW_CRITERIA_PATH="raw_criteria.jsonl"
EMBEDDED_CRITERIA_PATH="embedded_criteria.jsonl"
COMPRESSED_CRITERIA_PATH="compressed_criteria.txt"
VISUALIZATION_OUTPUT_PATH="figures/"
PATTERN_ANALYSIS_REPORT_PATH="pattern_analysis_report.jsonl"
RUBRIC_PATH="rubric.jsonl"
OPENAI_API_KEY="<your_openai_api_key>"
OPENAI_MODEL="gpt-4o"
VISUALIZE=True
```

If you want to directly generate the pattern report in one step, run the following command.
```bash
bash run.sh
```

To execute each step of CoT-Encyclopedia individually in a step-by-step manner, run the following commands.
### 1. Classification Criteria Identification
```bash
python classification_criteria_identification.py \
--dataset_path $DATASET_PATH \
--output_path $RAW_CRITERIA_PATH \
--openai_api_key $OPENAI_API_KEY \
--model $OPENAI_MODEL
```
### 2. Classification Criteria Embedding
```bash
python classification_criteria_embedding.py \
--input_path $RAW_CRITERIA_PATH \
--output_path $EMBEDDED_CRITERIA_PATH \
--openai_api_key $OPENAI_API_KEY
```
### 3. Classification Criteria Compression via Hierarchical Clustering
```bash
python hierchical_clustering.py \
--input_path $EMBEDDED_CRITERIA_PATH \
--compressed_criteria_path $COMPRESSED_CRITERIA_PATH \
--visualize $VISUALIZE \
--figure_output_path $VISUALIZATION_OUTPUT_PATH
```
### 4. Rubric Generation
```bash
python rubric_generation.py \
--compressed_criteria_path $COMPRESSED_CRITERIA_PATH \
--output_path $RUBRIC_PATH \
--api_key $OPENAI_API_KEY \
--model $OPENAI_MODEL
```
### 5. Pattern Analysis Report Generation
```bash
python pattern_analysis_report_generation.py \
--rubric_path $RUBRIC_PATH \
--dataset_path $DATASET_PATH \
--output_path $PATTERN_ANALYSIS_REPORT_PATH \
--api_key $OPENAI_API_KEY \
--model $OPENAI_MODEL
```
## Contact
Questions, pull requests, or error reports are welcome! `seongyun@kaist.ac.kr`
