# VasoPredict 🏥

> Explainable Early Vasopressor Decision Support Using MIMIC-IV ICU Data

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

VasoPredict is a clinical decision support system that predicts the early
need for vasopressor therapy in septic ICU patients. It combines a hybrid
**Deep Q-Learning + Grey Wolf Optimization (QGWO)** approach for adaptive
feature selection with explainable ML classifiers — enabling both high
predictive accuracy and transparent, clinician-readable reasoning.

---

## 🧠 What It Does

Septic shock is one of the leading causes of ICU mortality. Knowing *when*
to initiate vasopressors like norepinephrine is critical, but subtle early
physiological signals are easy to miss. VasoPredict:

- **Extracts** temporal clinical features (vitals, labs, medications) from MIMIC-IV
- **Selects** the most informative feature subset using the hybrid QGWO optimizer
- **Predicts** vasopressor initiation using LightGBM, XGBoost, and Random Forest
- **Explains** every prediction with SHAP, highlighting key drivers like
  lactate trends and blood pressure decline

---

## 📁 Project Structure
```
vasopredict/
├── data/
│   ├── raw/               # Original MIMIC-IV extracts (not tracked by Git)
│   └── processed/         # Cleaned, windowed feature tables (not tracked by Git)
├── notebooks/
│   └── healthcare_analysis.ipynb   # Full pipeline walkthrough
├── src/
│   ├── preprocessing.py            # Cohort extraction, imputation, normalization
│   ├── feature_engineering.py      # Temporal windows + statistical aggregates
│   ├── model.py                    # QGWO optimizer + classifier training
│   └── evaluation.py              # Metrics, SHAP analysis, visualizations
├── results/
│   ├── figures/                    # SHAP plots, ROC curves
│   └── metrics/                    # AUROC, F1, confusion matrices
├── models/                         # Saved model files
└── docs/                           # Report and architecture diagrams
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/vasopredict.git
cd vasopredict
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get the data

> ⚠️ MIMIC-IV data is **not included** in this repository.
> Access requires completing CITI training and signing a data use agreement
> at [PhysioNet](https://physionet.org/content/mimiciv/).

Once approved, place the raw CSV exports inside `data/raw/`.

---

## 🚀 Running the Pipeline

Run the full pipeline interactively via the notebook:
```bash
jupyter notebook notebooks/healthcare_analysis.ipynb
```

Or run each stage as a script:
```bash
python src/preprocessing.py        # Clean and impute raw data
python src/feature_engineering.py  # Generate temporal features
python src/model.py                # Run QGWO + train classifiers
python src/evaluation.py           # Evaluate and generate SHAP plots
```

---

## 🔬 Methodology

### Feature Engineering
Raw ICU time-series are aggregated over multi-hour temporal windows
(3h, 6h, 12h, 24h, 48h). Each window produces statistical features per
variable: **mean, min, max, standard deviation, and slope**.

### QGWO — Hybrid Feature Selection

We start with **150+ candidate features**. The QGWO algorithm finds the
optimal 25-feature subset in two layers:

**Grey Wolf Optimizer (GWO)** — inspired by wolf pack hunting behavior:
- 20 wolves represent 20 different feature subsets
- Alpha, Beta, and Delta wolves represent the top 3 subsets by AUC
- Omega wolves update their positions by following the leaders
- Runs for 30 iterations to converge on the best feature combination

**Deep Q-Network (DQN)** — fixes GWO's fixed exploration schedule:
- Observes state: features selected, current AUC, diversity, iteration progress
- Chooses actions: increase exploration / decrease exploration / maintain
- Learns when to explore broadly vs. refine around top-performing subsets

**Four key enhancements over standard GWO:**

| Enhancement | Description |
|---|---|
| Q-Learning Adaptive Convergence | DQN replaces fixed linear schedule |
| Segmented Position Updates | Exploration uses avg of top-4; exploitation uses weighted top-3 |
| Random Jumps | 1% chance per iteration to escape local minima |
| Bottom-30% Replacement | Worst wolves replaced with fresh random subsets each iteration |

### Classification
| Model | Description |
|---|---|
| LightGBM | Fast gradient boosting — best overall AUC |
| XGBoost | Robust, strong baseline |
| Random Forest | Stable, interpretable |

### Explainability
SHAP (SHapley Additive exPlanations) generates global feature importance
rankings and patient-level prediction breakdowns for clinical transparency.

---

## 🎯 Top Predictive Features (QGWO Selected)

| Rank | Feature | Importance | Clinical Significance |
|---|---|---|---|
| 1 | Lactate Max (6h) | 18% | Tissue hypoperfusion marker; critical >4.0 mmol/L |
| 2 | Blood Pressure Min (6h) | 15% | Shock indicator; vasopressor threshold <65 mmHg |
| 3 | Vasopressor Use | 12% | Indicates escalating care |
| 4 | Lactate Slope (6h) | 10% | **Trend matters more than level** — rising = worsening |
| 5 | Heart Rate Mean (6h) | 8% | Compensatory tachycardia; concern >110 bpm |
| 6 | Urine Output (6h) | 7% | Kidney perfusion; oliguria <200 mL |
| 7 | Respiratory Rate Max | 6% | Tachypnea >24 indicates distress |
| 8 | Temperature Mean | 5% | Less important than expected |
| 9 | WBC Max | 4% | Infection marker — less critical than hemodynamics |
| 10 | Fluid Balance | 3% | Resuscitation status |

**Key clinical insights from SHAP:**
- Lactate *dynamics* outperform static values — slope is a top-4 predictor
- Hemodynamics (BP, HR) matter more than inflammatory markers (WBC, temp)
- Confirms Sepsis-3 over SIRS: temperature and WBC rank low

---

## 📊 Model Performance

| Metric | Baseline (150 features) | QGWO (25 features) | Change |
|---|---|---|---|
| ROC-AUC | 0.81 | **0.87** | +7.4% ✅ |
| Sensitivity | 76% | **91%** | +15% ✅ |
| Specificity | 88% | 83% | −5% |
| F1 Score | 0.79 | **0.86** | +8.9% ✅ |
| Feature Count | 150 | **25** | −83% ✅ |

### Temporal Window Analysis

| Window | AUC | Notes |
|---|---|---|
| 3h | 0.79 | Too early — high noise, many false alarms |
| **6h** | **0.87** | ✅ Optimal — best balance of accuracy and timeliness |
| 12h | 0.84 | Slightly late |
| 24h | 0.81 | Patient often already decompensating |
| 48h | 0.78 | Treatment window likely missed |

### Clinical Impact
- Vasopressor initiation **2.3 hours earlier** than standard practice
- QGWO catches **15% more sepsis cases** (91% vs 76% sensitivity)
- Every hour of delay in vasopressor initiation increases mortality by ~7%
- 83% fewer features means faster real-time inference and clinician trust

---

## 📊 Dataset

- **MIMIC-IV v3.1** — Medical Information Mart for Intensive Care (MIT)
- 10,000 ICU patients (5,000 sepsis / 5,000 non-sepsis)
- Data components: laboratory results, vital signs, medication administration,
  output events, demographics, severity scores (SOFA, Charlson)

---

## 📚 Key References

- Mirjalili et al. (2014). Grey Wolf Optimizer. *Advances in Engineering Software.*
- Lundberg & Lee (2017). A unified approach to interpreting model predictions. *NeurIPS.*
- Johnson et al. (2023). MIMIC-IV. *Scientific Data.*
- Zhou et al. (2026). Early prediction of septic shock using ML. *Int. J. Medical Informatics.*
- Duval et al. (2025). Early prediction of vasopressor initiation. *BMC Medical Informatics.*
- Wu et al. (2023). Value-based deep RL for optimal sepsis treatment. *npj Digital Medicine.*
- Singer et al. (2016). Sepsis-3 Definitions. *JAMA.*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> ⚕️ **Disclaimer:** VasoPredict is a research prototype for educational purposes
> and is **not intended for direct clinical use** without prospective validation,
> regulatory approval, and clinical oversight. Always consult qualified medical
> professionals for patient care decisions.
