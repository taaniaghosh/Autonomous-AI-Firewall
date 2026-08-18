# AI-Powered Intrusion Detection System

Production-style intrusion detection workflow combining hybrid ML detection, event correlation, LLM reasoning, and response playbooks for actionable incident analysis.

## Overview

This project delivers an end-to-end IDS pipeline that can run on historical or live-derived network flow data.

Core capabilities:

- Hybrid detection using ensemble confidence scoring
- Incident-level correlation of sample alerts into grouped events
- LLM-assisted reasoning with automatic template fallback
- HoneyBadger response playbooks by severity
- Streamlit dashboard with exportable incident artifacts

## Current Architecture

The implemented runtime architecture is a 5-stage pipeline:

1. Hybrid Detection Engine
2. Correlation Engine
3. Reasoning Layer
4. Response Playbook (HoneyBadger)
5. Dashboard and Export Layer

### Hybrid Detection Model

Implemented detector components:

- LSTM proxy (MLP with temporal features)
- CNN proxy (MLP feature-pattern learner)
- Random Forest classifier
- Autoencoder-style reconstruction model for anomaly score

Scoring equations:

- Phybrid = 0.35 * PLSTM + 0.30 * PCNN + 0.35 * PRF
- Confidence = 0.72 * Phybrid + 0.28 * Anomaly

## Repository Structure

```text
intrusion_project/
  data/
    raw/
    processed/
      clean_data.csv
      engineered_features.csv
      multi_dataset_combined.csv
      cache/
  notebooks/
    1_data_preprocessing.ipynb
    2_feature_engineering.ipynb
    3_ml_training.ipynb
  intelligent_cyber_assistant.py
  live_data_adapter.py
  multi_dataset_processor.py
  streamlit_dashboard.py
  requirements.txt
  PROJECT_REPORT.md
  README.md
```

## Datasets

Supported sources:

- UNSW-NB15
- CIC-IDS2017 (CSV exports)

Combined processing is handled by multi_dataset_processor.py and produces:

- data/processed/multi_dataset_combined.csv

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

Set one or both providers in .env:

- OPENAI_API_KEY
- GEMINI_API_KEY (or GOOGLE_API_KEY)

### 3. Optional: build combined dataset

```bash
python multi_dataset_processor.py
```

### 4. Run backend pipeline

```bash
python intelligent_cyber_assistant.py
```

### 5. Run dashboard

```bash
python -m streamlit run streamlit_dashboard.py
```

## Streamlit Dashboard Features

- Historical CSV mode and real-time network mode
- Fast Mode toggle for lower-latency inference
- Cached preprocessing artifacts to speed repeated runs
- Session result cache for identical parameter reruns
- Heavy visual sections collapsed by default for faster first render
- JSON and CSV incident export

## LLM Reasoning Behavior

The Reasoning Layer supports OpenAI and Gemini providers.

Expected behavior:

- llm_used: yes when a provider call succeeds
- llm_used: no when fallback template reasoning is used

Fallback conditions include:

- Missing API keys
- Provider quota exhaustion or API errors
- Unsupported model identifier

Notes:

- In auto mode, OpenAI is primary when using GPT model names.
- Gemini fallback is available when Gemini key and valid model are configured.

## Performance Optimizations Implemented

- Feature-space reduction by excluding high-cardinality identity columns from training
- NaN and inf sanitization during dataset preparation
- float32 conversion for feature matrices
- Fast Mode detector configuration (smaller model sizes)
- On-disk preprocessed cache in data/processed/cache
- Streamlit-side result caching and optional cache clear

## Troubleshooting

### Streamlit starts but analysis appears slow

- Enable Fast Mode in sidebar
- Lower sample size (for example 2000 to 4000)
- Use Clear cached results only when you need a fresh run

### LLM card shows llm_used: no

- Verify API key exists in .env
- Ensure selected provider matches configured key
- Check model name is supported
- Clear cached results and rerun

### Model errors about NaN values

This is handled in current preprocessing logic. If seen again, regenerate the combined dataset and rerun with a lower sample size to isolate corrupted input slices.

## Security Notes

- Never commit .env to source control
- Rotate API keys if exposed in logs, screenshots, or chat
- Keep raw and processed datasets under local trusted storage

## Tech Stack

- Python, Pandas, NumPy
- scikit-learn
- Streamlit
- Matplotlib, Seaborn
- python-dotenv
- OpenAI SDK and Google Generative AI SDK

## License and Use

This repository is currently configured as an academic and research-style implementation. Add an explicit license file if you plan broader public distribution.
*Structured, Complete, Production-Ready*
