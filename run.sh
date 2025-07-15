#!/bin/bash

# ====== 1. Classification Criteria Identification ======
python classification_criteria_identification.py \
  --dataset_path "$DATASET_PATH" \
  --output_path "$RAW_CRITERIA_PATH" \
  --openai_api_key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

# ====== 2. Classification Criteria Embedding ======
python classification_criteria_embedding.py \
  --input_path "$RAW_CRITERIA_PATH" \
  --output_path "$EMBEDDED_CRITERIA_PATH" \
  --openai_api_key "$OPENAI_API_KEY"

# ====== 3. Classification Criteria Compression via Hierarchical Clustering ======
python hierchical_clustering.py \
  --input_path "$EMBEDDED_CRITERIA_PATH" \
  --compressed_criteria_path "$COMPRESSED_CRITERIA_PATH" \
  --visualize "$VISUALIZE" \
  --figure_output_path "$VISUALIZATION_OUTPUT_PATH"

# ====== 4. Rubric Generation ======
python rubric_generation.py \
  --compressed_criteria_path "$COMPRESSED_CRITERIA_PATH" \
  --output_path "$RUBRIC_PATH" \
  --api_key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"

# ====== 5. Pattern Analysis Report Generation ======
python pattern_analysis_report_generation.py \
  --rubric_path "$RUBRIC_PATH" \
  --dataset_path "$DATASET_PATH" \
  --output_path "$PATTERN_ANALYSIS_REPORT_PATH" \
  --api_key "$OPENAI_API_KEY" \
  --model "$OPENAI_MODEL"