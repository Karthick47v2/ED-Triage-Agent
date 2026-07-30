"""Shared ESI resource-prediction rubric used by triage prompts."""

from textwrap import dedent


ESI_RESOURCE_RUBRIC_SECTION = dedent(
    """  
    ### ESI Levels 3–5: Resource Prediction  
  
    Apply this rubric only after ruling out ESI-1 and ESI-2.  
    Resource count must never justify ESI-1 or ESI-2.  
  
    #### Counting  
  
    Predict the number of distinct resource types likely required during  
    this ED visit. Count each type once, regardless of order quantity.  
  
    #### Resources: 1 point each  
  
    - Labs: all blood and urine tests combined  
    - ECG  
    - X-rays  
    - CT, MRI, or ultrasound  
    - IV fluids  
    - IV, IM, or nebulized medications  
    - Specialty consultation, including Social Work or Psychiatry  
    - Simple procedure, such as laceration repair, Foley catheterization,  
      or joint reduction  
  
    #### Not resources: 0 points  
  
    - History and physical examination, including pelvic examination  
    - Point-of-care testing  
    - Saline lock without IV fluids or medication  
    - Oral medication  
    - Tetanus immunization  
    - Prescription writing or refill  
    - Phone call to a primary-care provider  
    - Simple wound care  
    - Crutches, splints, or slings  
    - Simple nursing interventions  
  
    #### ESI determination  
  
    - 0 resources: ESI-5  
    - 1 resource: ESI-4  
    - 2 or more resources: ESI-3  
  
    #### Canonical examples  
  
    - ESI-5: Cold symptoms requiring bulb suction or oral medication;  
      prescription refill only.  
    - ESI-4: Ankle sprain requiring an X-ray; simple laceration requiring  
      repair; nursemaid's elbow requiring reduction.  
    - ESI-3: Abdominal pain requiring labs and imaging; pregnancy requiring  
      ultrasound and Social Work consultation.  
  
    #### Ambiguity  
  
    When the expected resource count is unclear:  
  
    1. State the uncertainty.  
    2. Identify the closest canonical example.  
    3. State the assumptions used.  
    4. Under partial Phase 1 information, avoid under-triage.  
    """
).strip()
