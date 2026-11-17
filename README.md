# Patient_readmission_case_study


Predicting 30-Day Patient Readmission
A supervised machine-learning case study exploring whether historical hospital admission data can help predict whether an admission will be followed by readmission within 30 days of discharge.
Project status: Educational case study. The current tuned Random Forest performed poorly on the held-out test set and is not suitable for clinical use.

Project overview
- Problem: Estimate whether an admission will be followed by readmission within 30 days.
- Prediction point: Discharge.
- Environment: Databricks, Python and scikit-learn.
- Dataset: 3,000 admission records.
- Target: Readmission (Yes / No).
- Class balance: 2,134 No (71.13%) and 866 Yes (28.87%).
Dataset fields
The case-study data dictionary includes Patient ID, Age, Gender, Admission Type, Length of Stay, Number of Diagnoses, Blood Pressure, Blood Sugar Levels, Previous Admissions and Readmission. Patient ID is an admission-record identifier and was excluded from predictors.

The notebook also derives Age Group and Stay Category for exploratory analysis.
Workflow
1. Data loading and inspection
- Read the Spark table patient.readmission.patient_readmission_data.
- Convert the Spark DataFrame to Pandas.
- Inspect shape, sample records and descriptive statistics.
- Check exact duplicate rows and target categories.
- Exclude the identifier from model predictors.
2. Data preparation
- Strip surrounding spaces from text values and standardise target labels.
- Create a separate modelling DataFrame.
- Use median imputation and standard scaling for numeric features.
- Use most-frequent imputation and one-hot encoding for categorical features.
- Place preprocessing inside scikit-learn pipelines to reduce data leakage.
Reproducibility note: the notebook should explicitly display missing-value counts by column and use one consistently defined cleaned dataset. The earlier dropna() copy was not consistently used downstream.
3. Exploratory data analysis
EDA included gender and admission type distributions, age and length-of-stay groups, diagnosis counts, readmission balance, readmission counts by gender, numeric distributions, box plots by target, readmission rates by gender and admission type, and a numeric correlation heatmap.

Key observations:
- Readmissions represented 28.87% of records.
- Readmission rates were 29.52% for male and 28.17% for female admissions.
- Rates were 29.37% for elective and 28.32% for emergency admissions.
- Mean diagnosis counts and previous-admission counts were similar across the target groups.
These are descriptive associations, not evidence of causation.
4. Train/test split and preprocessing
- 80% training data (2,400 records) and 20% test data (600 records).
- Stratified split with random_state=42.
- Patient ID excluded.
- Imputers, encoders and scalers fitted within pipelines.
5. Model development
Compared:
- Dummy Classifier baseline
- Logistic Regression
- Decision Tree
- Random Forest
Five-fold stratified cross-validation was applied to training data. Logistic Regression and Random Forest were tuned using GridSearchCV.
Model	Mean CV ROC-AUC	CV ROC-AUC SD	Mean accuracy
Random Forest	0.522	0.011	0.701
Decision Tree	0.517	0.018	0.601
Dummy baseline	0.500	0.000	0.711
Logistic Regression	0.492	0.025	0.486


Tuned Random Forest parameters: max_depth=20, min_samples_leaf=1; best recorded CV ROC-AUC: 0.524.
6. Final evaluation
Metric	Test score
Accuracy	0.703
Precision (Yes)	0.273
Recall (Yes)	0.017
F1-score (Yes)	0.033
ROC-AUC	0.458


Training ROC-AUC was 1.000 versus 0.458 on the test set. This is a large generalisation gap and indicates severe overfitting. Despite 70.3% accuracy, the model identified only 1.7% of actual readmissions at the default 0.50 threshold.
Limitations and required follow-up
1. Cross-validation scoring bug: precision, recall and F1 were NaN because the scorer expected positive label 1, while the target labels are strings (No, Yes). Configure scoring with pos_label="Yes" and rerun.
2. Feature-selection task incomplete: the notebook reports Random Forest feature importance but does not include the full statistical tests, effect sizes, mutual information, redundancy review and feature keep/drop table required by the assessment brief.
3. Threshold analysis: the notebook calculates threshold metrics using held-out test labels. Select the threshold using training-only out-of-fold predictions, then evaluate once on an untouched test set.
4. Potential redundant predictors: Age and Age Group, and Length of Stay and Stay Category, contain overlapping information. Review whether both forms are needed.
5. Clinical validation: this teaching dataset is not sufficient to establish clinical usefulness. Do not use the current model for patient-care decisions.
Tools and libraries
- Python
- Databricks / Apache Spark
- Pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
Suggested repository structure
patient-readmission-prediction/
├── README.md
├── notebooks/
│   └── Patient_Readmission_Notebook.ipynb
├── reports/
│   └── Patient_Readmission_Report.pdf
└── assets/
    └── figures/
Do not upload identifiable patient information or confidential data. Publish the dataset only if you have permission and it is appropriately anonymised.
Reproducing the analysis
1. Open the notebook in a Databricks workspace with access to the source table.
2. Confirm the source table and column names.
3. Run the notebook from top to bottom.
4. Review warnings and verify that all metric outputs are valid.
5. Correct the feature-selection and threshold-validation gaps listed above.
6. Update the report so every number matches the final notebook.
References
- BrightLearn (2026), Machine Learning Case Study: Predicting 30-Day Patient Readmission, Student Brief.
- Pedregosa, F. et al. (2011), 'Scikit-learn: Machine Learning in Python', Journal of Machine Learning Research, 12, pp. 2825–2830.
- scikit-learn documentation: https://scikit-learn.org/stable/
- pandas documentation: https://pandas.pydata.org/docs/
Academic integrity
This README describes the workflow and results recorded in the notebook. Update it if the analysis changes. Declare AI tools, tutorials, external sources and collaborators according to the assessment requirements.
