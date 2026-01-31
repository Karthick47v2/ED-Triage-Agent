# ED-Triage-Agent

Official implementation for **ED-Triage-Agent: A Framework for Human-AI Collaborative Emergency Triage**.

This repository contains the code for the two-phase ED triage pipeline: **Phase 1** (symptom-based: IIA -> CRA -> PAA) and **Phase 2** (post-vital: CRA Phase 2 -> TCA) with deterministic vital-sign assessment and ESI-aligned classification.

---

## Installation

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your Azure OpenAI credentials.

**RAG (Phase 1):** Ingest the ESI Handbook into ChromaDB once:

```bash
python -m ed_triage.cra.rag_setup --pdf ESI_Handbook.pdf --reset
```

---

## Reproducing Results

All commands are run from the **repository root**. Scenario data and extracted vitals/physical exam files are in `eval/`.

### Phase 1 (IIA -> CRA -> PAA)

Run all scenarios and save results + summary:

```bash
python eval/run_evaluation.py --scenarios-file eval/scenarios_with_answers.json --output results/phase1_results.json --all
```

Summary and metrics are written to `results/phase1_results_summary.json`. Optional flags: `--verbose`, `--limit N`, `--scenario N` (single scenario).

### Phase 2 (CRA Phase 2 -> TCA)

Requires Phase 1 results and the provided `eval/vital_signs_extracted.json` and `eval/physical_exam_extracted.json`.

```bash
python eval/run_phase2_evaluation.py \
  --phase1-results results/phase1_results.json \
  --vital-signs eval/vital_signs_extracted.json \
  --physical-exam eval/physical_exam_extracted.json \
  --output results/phase2_results.json
```

Phase 2 prints ESI exact-match and within-±1 accuracy to the console. Use `-v` for verbose output or `--limit N` to run on a subset of scenarios.

---

## Interactive Demo

Full Phase 1 pipeline (IIA chat -> CRA -> PAA):

```bash
python -m ed_triage.main
```

IIA-only (intake interview + extraction):

```bash
python -m ed_triage.iia.main
```

<!-- ---

## Citation

If you use this code or the paper in your work, please cite:

```bibtex
@article{edtriage2025,
  title={ED-Triage-Agent: A Framework for Human-AI Collaborative Emergency Triage},
  author={...},
  year={2025}
}
``` -->
