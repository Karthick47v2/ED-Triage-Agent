# ED-Triage-Agent

Implementation of **ED-Triage-Agent: A Framework for Human-AI Collaborative Emergency Triage**.

## Architecture

The system is a two-phase pipeline of cooperating LLM agents:

- **Phase 1 — Pre-vitals queue priority.** `IIA -> CRA -> PAA`
  - IIA conducts an OLDCARTS intake interview and extracts a structured `IntakeSummary`.
  - CRA reasons over the intake against ESI-handbook passages retrieved via RAG.
  - PAA emits a tentative ESI level (1–5) and a HIGH/LOW queue priority.
- **Phase 2 — Final ESI after vitals + exam.** `CRA(phase2) -> TCA`
  - CRA re-runs with vital signs and physical exam findings.
  - TCA produces the final ESI recommendation using a deterministic vital-signs
    assessment plus age-aware ESI handbook thresholds (Tables 6-1, 6-2, Figure 6-1).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill `.env` with Azure OpenAI credentials. By default the system uses GPT-4.1 for CRA/PAA/TCA and GPT-4.1-mini for the IIA and the simulated patient agent.

Build the CRA retrieval index once:

```bash
python -m ed_triage.cra.rag_setup --pdf ESI_Handbook.pdf --reset
```

## Interactive runs

Phase 1 triage loop:

```bash
python -m ed_triage.main
```

## Reproducing the evaluation

### Phase 1 (IIA -> CRA -> PAA, with simulated patient)

```bash
python -m eval.run_evaluation \
  --scenarios-file eval/practice_cases.json \
  --output results/phase1_practice_case_results.json \
  --all --patient-profile 1
```

### Phase 2 (CRA Phase 2 -> TCA, with pre-extracted vitals + exam)

```bash
python -m eval.run_phase2_evaluation \
  --phase1-results results/phase1_practice_case_results.json \
  --vital-signs eval/practice_cases_vital_signs_extracted.json \
  --physical-exam eval/practice_cases_physical_exam_extracted.json \
  --output results/phase2_practice_case_results.json
```

### Metrics and figures

```bash
# All Phase 1 metrics
python -m eval.metrics results/phase1_practice_case_results.json --phase 1

# All Phase 2 metrics, plus a confusion-matrix heatmap and JSON dump
python -m eval.metrics results/phase2_practice_case_results.json --phase 2 \
  --heatmap figures/cm_phase2.png \
  --json-out results/phase2_metrics.json

# Reliability diagram (calibration)
python -m eval.plot_reliability_diagram \
  --input results/phase1_practice_case_results.json \
  --output figures/reliability_phase1
```