# Bounded LLM audit of acoustic leak classifiers

Code for the study "Multi-model evaluation of bounded large language model auditors for explainable acoustic leak triage in water distribution networks".

An explained random forest scores each acoustic logger deployment. A large language model (LLM) reviews deterministic acoustic evidence and the
forest explanation, and may shift the forest log-odds by at most two steps of 0.7 (`p' = sigmoid(logit(p_RF) + 0.7a)`, `a` in {-2,...,+2}).
It also returns a rationale, an action recommendation (dispatch, extend monitoring or hold excavation) and a work order.

## Contents
- `pipeline/acoustic_features.py`: recording and deployment features (Welch spectra, band fractions, leak-band ratio, persistence, transient ratio).
- `pipeline/v3_02_packets.py`, `v3_03_inputs.py`, `v5_01_h2_inputs.py`: leave-one-site-out evidence packets (reference statistics, retrieval, random forest tool, reliability table, local effects), built from training sites only and anonymised.
- `pipeline/v3_05_benchmark.py`, `v5_03_benchmark.py`, `v6_score_blind.py`, `v12_01_assemble.py`: comparator models, scoring, site-level bootstrap, paired tests.
- `pipeline/v7_external_hybrids.py`, `v8_sheffield_local.py`, `v10_opus_external.py`: grouped evaluations on the two public datasets.
- `pipeline/v12_02_figures.py`, `p6figlib.py`, `v12_03_export.ps1`: editable PowerPoint figures.
- `public_data/`: download and feature extraction for the public Dongguan (Zenodo, doi:10.5281/zenodo.18631450) and Sheffield (doi:10.15131/shef.data.32229270.v2) datasets, both CC BY 4.0.
- `prompts/`: the exact audit prompts for the field and public-data evaluations and the zero-shot comparator prompt.

## Data
The field recordings, site identities, labels, deployment features and model outputs belong to the water utility project and are not included.
They cannot be shared without the utility's permission. Scripts expect them under `derived/` and a private `private/sites.json`.
The public datasets can be downloaded with `public_data/download_data.py`.

## Keys
LLM calls read `POE_API_KEY` and `GEMINI_API_KEY` from the environment. No key is stored in this repository.
