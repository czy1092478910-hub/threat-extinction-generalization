# Threat extinction & generalization model

This repository contains a minimal computational model of
**extinction failure and threat generalization**, inspired by
phenomena observed in anxiety disorders and PTSD.

---

## Research question
Why do some individuals fail to learn safety after extinction
(US no longer occurs), and continue to generalize threat to
similar but safe cues (GS)?

---

## Model overview
We implement a Rescorla–Wagner learning model with:

- Threat expectation value V
- Binary outcome r ∈ {0, 1}
- Prediction error: δ = r − V
- **Asymmetric learning rates**
  - α⁺ for threat learning (δ > 0)
  - α⁻ for safety learning (δ < 0)
- **Threat generalization**
  - Learning at CS+ spreads to GS with strength g

Impaired safety learning (low α⁻) naturally produces
extinction failure and persistent threat generalization.

---

## Files
- `simulate_rw.py`  
  Basic threat learning and extinction (CS only)

- `generalization.py`  
  Threat generalization from CS+ to GS under impaired safety learning

---

## How to run
```bash
python simulate_rw.py
python generalization.py

Key results

Reduced safety learning rate (α⁻) leads to extinction failure

Increased generalization strength (g) leads to elevated threat
responses to GS even after extinction

The interaction of α⁻ and g captures core features of anxiety/PTSD

Status

This is an ongoing exploratory project.
The model is intended as a minimal mechanistic demonstration
rather than a finalized cognitive model.

Contact

Prepared by: Chao Zhiyu
University of Szeged
