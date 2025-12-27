# PulseNet — Problematic Internet Use Predictor
Problematic Internet Use (PIU) among kids and
teenagers is a growing problem in today’s digital age, contributing
to mental health issues like anxiety and sadness. With the tech-
driven environment completely encapsulating us, it is essential to
recognize and treat problematic internet use early on to promote
mental health and prevent long-term psychological repercussions.
This project aims to develop machine learning models that predict
the Severity Impairment Index (sii), which measures problematic
internet use, by leveraging physical activity and internet usage
data from the Healthy Brain Network (HBN) as a part of the
Kaggle Competition, “Child Mind Institute – Problematic Inter-
net Use”. The dataset includes accelerometer (actigraphy) data,
fitness assessments, and various behavioral measures collected
from a Questionnaire. This project focuses on implementing
four machine learning models: Random Forest, Support Vector
Machines (SVM), Logistic Regression, and Clustering on this
data. Each model will analyze the relationship between physical
activity and problematic internet use to identify early indicators
of compulsive or excessive digital behavior. By comparing the
performance of these models based on metrics such as accuracy,
precision, and recall, this project will determine the most effective
approach for predicting problematic internet use.


## Repository layout

- EDA/
  Exploratory notebooks and analysis (visualizations, clustering experiments, correlations).
- Processing/
  Data cleaning and feature engineering scripts. Main script: [Clean_CSV_Train.py](cci:7://file:///c:/Users/samhi/Downloads/PulseNet-Problematic-Internet-Use-Predictor-main/Clean_CSV_Train.py:0:0-0:0).
- Model Implementations/
  Trained model artifacts (`.joblib`) and model-specific assets (Random Forest, XGBoost, KMeans, Ensembles).
- Data/
  CSVs and other data files used by notebooks and scripts.

## Processing pipeline (Processing/Clean_CSV_Train.py)

- Inputs
  - Reads two tables (`cleaned.csv`, `train.csv`) and merges on `id`.
  - Note: The script currently references absolute Windows paths; change them to use files under [Data/](cci:7://file:///c:/Users/samhi/Downloads/PulseNet-Problematic-Internet-Use-Predictor-main/Data:0:0-0:0).

- Column pruning
  - Drops survey “Season” fields, per-item PCIAT questions, totals/summary columns, and non-essential vitals (e.g., BP, waist).
  - Removes additional BIA and Fitness Endurance columns to simplify features.

- Merge & duplicate handling
  - Left-join on `id` to keep rows from the main table.
  - If columns duplicate with `_x`/`_y`, keep `_x`, drop `_y`, then strip `_x` suffix.

- Missing value imputation
  - KNN (k=5) for:
    - `CGAS-CGAS_Score`
    - `Normalized_Combined_Score` (later renamed to `Fitness_Combined_Score`)
    - `Physical_Composite_Index`
    - `Physical-BMI`
  - Mode for `Basic_Demos-Enroll_Season`.

- FitnessGram feature: `Fitness_Combined_Score`
  - Converts specific 3-category zone columns to binary for two metrics then drops them.
  - Sums scores of: `FGC-FGC_CU_Zone`, `FGC-FGC_PU_Zone`, `FGC-FGC_SRL_Zone`, `FGC-FGC_SRR_Zone`, `FGC-FGC_TL_Zone` (only if none are NaN).
  - Min-max normalizes to `Normalized_Combined_Score`, imputes (KNN), and renames to `Fitness_Combined_Score`.
  - Correlates with `sii` for sanity checks.

- BIA composite: `Physical_Composite_Index`
  - Weighted, min-max normalized composite of:
    - BIA_BMR (0.25), BIA_DEE (0.25), BIA_SMM (0.15), BIA_Fat (0.15), BIA_BMI (0.10),
    - BIA_TBW (0.05), BIA_ICW (0.05), BIA_ECW (0.05), BIA_LST (0.025), BIA_LDM (0.025)
  - KNN imputation (k=5) for missing values.
  - Validates correlations (Pearson, Spearman, Kendall), Mutual Information, and Distance Correlation.
  - Drops individual BIA columns afterwards.

- Activity level imputation with distribution preservation
  - For `BIA-BIA_Activity_Level_num`, imputes missing values to match original distribution:
    - 1.0: 12.995%, 2.0: 31.7903%, 3.0: 34.9136%, 4.0: 15.8951%, 5.0: 4.4060%
  - Uses a “physical score” (mean of min-max normalized `Physical-BMI`, `Fitness_Combined_Score`, `Physical_Composite_Index`) to sort and assign values.
  - Validates via pre/post distributions and correlations, then renames to `BIA_Activity_Level`.

- Final schema and output
  - Ordered columns exported:
    - `id`, `Basic_Demos-Age`, `Basic_Demos-Sex`, `Basic_Demos-Enroll_Season`, `age_group`,
    - `CGAS_Score`, `Physical-BMI`, `BIA_Activity_Level`, `Fitness_Combined_Score`,
    - `Physical_Composite_Index`, `SDS-SDS_Total_T`, `PreInt_EduHx-computerinternet_hoursday`, `sii`
  - Saves to [cleaned_train.csv](cci:7://file:///c:/Users/samhi/Downloads/PulseNet-Problematic-Internet-Use-Predictor-main/cleaned_train.csv:0:0-0:0) (move to [Data/](cci:7://file:///c:/Users/samhi/Downloads/PulseNet-Problematic-Internet-Use-Predictor-main/Data:0:0-0:0) if preferred).

## Model implementations

- Artifacts (examples)
  - `Model Implementations/XGBoost_model_pipeline.joblib` — pipeline for SII prediction.
  - `Model Implementations/kmeans_model_pipeline_with_visualization.joblib` — clustering pipeline.
  - `Model Implementations/Ensemble_model_pipeline.joblib` — meta-ensemble over base models.
  - `Model Implementations/Models/Random Forest/` — tuned Random Forest models and preprocessors (e.g., PCA, scaler).

- Loading a trained pipeline
```python
import joblib, pandas as pd
df = pd.read_csv("Data/cleaned_train.csv")  # ensure columns match training
model = joblib.load("Model Implementations/XGBoost_model_pipeline.joblib")
preds = model.predict(df)
print(preds[:10])
