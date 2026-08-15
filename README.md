# Mission Health — Early Risk Prediction of Type-2 Diabetes in Under-Resourced Clinics

CIA-3 Machine Learning: ML for Social Good Challenge (Mission: Health)

## 1. Problem

**Screen adult patients in under-resourced primary-care clinics for early Type-2 diabetes risk using routinely collected health measurements, helping frontline healthcare workers prioritise patients for confirmatory testing and follow-up.**

This system is an early-risk screening/decision-support prototype, NOT a diagnostic system. It is intended to support, not replace, healthcare professionals.

## 2. Dataset

- **Name:** Pima Indians Diabetes Database
- **Original source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- **Availability:** Publicly available through the UCI Machine Learning Repository / Kaggle.
- **File used:** `data/diabetes.csv` — 768 records, 8 input features, 1 binary outcome. No names, IDs, or direct personal identifiers are present.
- **Citation:** National Institute of Diabetes and Digestive and Kidney Diseases. *Pima Indians Diabetes Database.* UCI Machine Learning Repository / Kaggle. Retrieved via `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`, `https://www.kaggle.com/datasets/jamaltariqcheema/pima-indians-diabetes-dataset` (dataset link).

The source cohort consists of adult women of Pima Indian heritage, so the model must not be assumed to generalise to other populations without external validation.

## 3. Project structure

```text
mission_health/
├── codebase/Mission_Health_CIA3.ipynb
├── data/diabetes.csv
├── results/
├── requirements.txt
└── README.md
```

## 4. Reproducibility

To reproduce the environment and run the notebook:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace Mission_Health_CIA3.ipynb
```

Alternatively, the notebook can be opened in Jupyter Notebook, JupyterLab, or VS Code and executed using **Run All**.

- `RANDOM_STATE = 42` is fixed throughout the notebook.
- The train/validation/test splitting is reproducible.
- Cross-validation uses a fixed random state.
- Model random states are fixed where applicable.

## 5. Methodology

### Data preprocessing
- Detect duplicate records.
- Audit data types.
- Identify biologically invalid zero values.
- Convert invalid zero values in:
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  to missing values.
- Handle missing values through the preprocessing pipeline.
- Scale numerical features where required.
- Perform preprocessing without data leakage.

### Class imbalance
The dataset has approximately:
- 65.1% negative class
- 34.9% positive class

Class imbalance is handled using class weighting. Logistic Regression, Decision Tree and Random Forest use `class_weight='balanced'`. XGBoost uses `scale_pos_weight` calculated from the training data.

### Feature engineering
- BMI category
- Glucose category
- Age group
- Glucose × BMI interaction

### Data split
The data is divided into stratified training, validation and test sets using a 60% / 20% / 20% split.

### Leakage prevention
Preprocessing and model tuning are performed without using the test set. Imputation and scaling are fitted within the training workflow and are not fitted using validation or test data.

### Model selection
Cross-validation and hyperparameter tuning are performed using the training data. The validation set is then used to select the final model. The test set remains untouched until final evaluation.

## 6. Models Evaluated

### Baselines
- Decision Tree
- Logistic Regression

### Bagging
- Random Forest

### Boosting
- XGBoost

### Heterogeneous ensembles
- Stacking: Logistic Regression + Random Forest + XGBoost
- Soft Voting: Logistic Regression + Random Forest + XGBoost

Cross-validation and hyperparameter tuning were used where applicable.

## 7. Final Model Selection

XGBoost was selected as the final model based on validation performance, achieving a validation ROC-AUC of **0.9437**.

The test set was not used for model selection. After XGBoost was selected using validation performance, it was evaluated on the untouched test set.

**Final XGBoost test ROC-AUC: 0.9282**

## 8. Final Test-Set Results

**Final test-set results on the untouched test split**

| Model                         |    ROC-AUC |   F1-Score | Precision |     Recall |
| ----------------------------- | ---------: | ---------: | --------: | ---------: |
| Baseline: Decision Tree       |     0.8820 |     0.7678 |    0.7288 |     0.8113 |
| Baseline: Logistic Regression |     0.8342 |     0.7213 |    0.6376 |     0.8301 |
| Bagging: Random Forest        |     0.9164 |     0.8301 |    0.8301 |     0.8301 |
| Boosting: XGBoost             | **0.9282** |     0.8301 |    0.8301 |     0.8301 |
| Stacking (LR + RF + XGB)      |     0.9187 |     0.8333 |    0.8181 |     0.8490 |
| Voting (LR + RF + XGB)        |     0.9065 | **0.8440** |    0.8214 | **0.8679** |

## 9. Results Interpretation

XGBoost achieved the highest test ROC-AUC at 0.9282, indicating the strongest overall ranking performance among the evaluated models.

Voting achieved the highest F1-score (0.8440) and recall (0.8679), while Stacking achieved a recall of 0.8490.

Because this is an early-risk screening problem, recall is an important metric because false negatives represent potentially at-risk patients who may not be flagged for confirmatory testing. However, increasing recall can also increase false positives, so the operating threshold should ultimately be determined with clinical expertise and deployment-specific validation.

## 10. Explainability

SHAP explainability was applied to the actual validation-selected XGBoost model.

### Global explanation
`outputs/shap_global_summary.png`
The global SHAP summary plot highlights the most influential features driving predictions, including Glucose, BMI, Age, and the engineered Glucose × BMI interaction.

### Local explanation
`outputs/shap_local_explanation.png`
This plot shows the contribution of individual features for one held-out test record.

### Synthetic demonstration
`outputs/shap_synthetic_patient_demo.png`
This is a separate synthetic record created specifically for the live demonstration. It does not represent a real patient from the test set.

## 11. Ethics Statement

**This project is an academic machine-learning screening prototype, not a clinically validated diagnostic system.**

- **Population bias / External validation:** The dataset's population limitations (single-cohort, geographic region) mean the model should not be deployed to other populations without appropriate external validation.
- **Privacy:** While the dataset is public and de-identified, real deployment requires health-data protection compliance.
- **Uncertainty & False positives/negatives:** The model outputs probabilities that should inform a triage threshold balancing the cost of missed cases (false negatives) against unnecessary testing (false positives).
- **Fairness & Human oversight:** The system must be monitored for fairness across subgroups, and all final medical decisions require human oversight.

## 12. Academic Integrity

The project uses standard publicly documented libraries including pandas, scikit-learn, XGBoost, SHAP and related dependencies. Their use is acknowledged in the project requirements and code.
