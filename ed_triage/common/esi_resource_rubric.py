"""Shared ESI levels 3–5 resource-prediction rubric for CRA, PAA, and TCA prompts."""

ESI_RESOURCE_RUBRIC_SECTION = """
### ESI LEVELS 3–5 — RESOURCE PREDICTION

Apply ONLY after ESI-1 and ESI-2 criteria have been ruled out. Resource counting determines ESI 3, 4, or 5 only. Never use resource count to justify ESI-1 or ESI-2.

#### HOW TO COUNT RESOURCES
Predict the number of distinct resource TYPES this ED visit will likely require. Do not count the same resource type more than once regardless of quantity ordered.

#### WHAT COUNTS AS A RESOURCE — 1 Point Each
- Labs (blood, urine) — all lab tests combined count as 1 resource
- ECG
- X-rays
- CT, MRI, or ultrasound
- IV fluids
- IV, IM, or nebulised medications
- Specialty consultation (including Social Work, Psychiatry)
- Simple procedure (e.g. laceration repair, Foley catheter, joint reduction)

#### WHAT DOES NOT COUNT AS A RESOURCE — 0 Points
- History and physical examination (including pelvic exam)
- Point-of-care testing (bedside glucose, urine dipstick, rapid pregnancy test, rapid Strep test)
- Saline lock (unless IV fluids or medications are administered)
- Oral medications
- Tetanus immunisation
- Prescription writing or refills
- Phone call to primary care provider
- Simple wound care (cleaning, basic dressings, suture removal)
- Crutches, splints, slings
- Simple nursing interventions (nasal bulb suctioning, ice pack, warming blanket)

#### ESI LEVEL DETERMINATION
- 0 resources expected → ESI-5
- 1 resource expected → ESI-4
- 2 or more resources expected → ESI-3

#### CANONICAL EXAMPLES (ESI Handbook v4)
ESI-5: Cold symptoms (bulb suctioning = 0, oral meds = 0); prescription refill only.

ESI-4: Ankle sprain needing X-ray (X-ray = 1, splint = 0); simple laceration needing suture repair (suture = 1); classic nursemaid's elbow needing reduction (reduction = 1).

ESI-3: Abdominal pain needing labs and imaging (labs = 1, CT = 1 = 2 total); pregnant patient needing ultrasound and social work (ultrasound = 1, social work = 1 = 2 total).

#### AMBIGUITY RULE
When resource count is unclear, state your uncertainty explicitly in reasoning. Align with the closest canonical example above and state the assumption made. Under partial information (Phase 1), prefer conservative priority when resource count is uncertain.
""".strip()
