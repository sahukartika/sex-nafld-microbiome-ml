# Interplay of Sex and NAFLD in Microbial Signatures: A Machine Learning Perspective

This repository supports our manuscript submitted to *Academia Nutrition and Dietetics*. We investigated gut microbiome patterns in relation to sex and non-alcoholic fatty liver disease (NAFLD) using a machine learning approach. The repository includes:

- Preprocessing scripts to convert MSP-level abundances to phylum level
- Scripts for binary classification tasks (disease and sex)
- SHAP analysis to interpret microbial drivers

---

## 🔗 Data Access

Due to file size limitations, only the custom MSP-to-phylum mapping file (`mspmap.xlsx`) is included in this repository.

Please manually download the following files from the [Human Gut Microbiome Atlas](https://www.microbiomeatlas.org):

- `vect_atlas.csv` — Abundance matrix (MSPs as rows, samples as columns)
- `sampleID.csv` — Sample metadata including `sample.ID`, `Gender`, and `Disease`

After downloading, place them in the root directory of this repository before running any scripts.


---

## 📁 File Overview

| File                              | Description                                                   |
|-----------------------------------|---------------------------------------------------------------|
| `01_msp_to_phylum_aggregation.py` | Aggregates MSPs to phylum-level abundance                     |
| `02_groupwise_split.py`           | Extracts and saves phylum data by disease and sex groups      |
| `model_comparison.py`             | Compares classifiers (XGBoost, RF, GB, Extra Trees)           |
| `finalmodel.py`                        | Final modeling and SHAP value interpretation                  |
| `mspmap.xlsx`                     | Mapping of MSPs to their corresponding phylum                 |
| `README.md`                       | Instructions for reproducing the results                      |


---

## ⚙️ Step-by-Step Instructions

### Step 1: Preprocessing — Aggregate MSPs to Phyla

```bash
python 01_msp_to_phylum_aggregation.py
*From output, manually remove the non phylum categories. They come under unclassified and unclassified eukaryota in Human Gut Microbiome Atlas. 
---

```markdown
```

###  Step 2: Groupwise Data Extraction

```bash
python 02_groupwise_split.py
```

* This uses `sampleID.csv` and the output from Step 1
* Generates `phylum_abundance_by_group.xlsx`, which contains separate sheets for:

  * `Healthy_male`
  * `Healthy_female`
  * `NAFLD_male`
  * `NAFLD_female`

These files are used for group-specific classification tasks.

---

## Step 3: Model Comparison

```bash
python model_comparison.py
```

* Runs classification tasks on three scenarios:

  1. NAFLD vs. Healthy
  2. Male vs. Female in Healthy group
  3. Male vs. Female in NAFLD group

* Compares four models:

  * XGBoost
  * Random Forest
  * Gradient Boosting
  * Extra Trees

* Outputs performance metrics (precision, recall, F1, AUC)

* Saves comparative bar plots of model performance (F1 Score)

---

## Step 4: Final Model and SHAP Interpretation

```bash
python finalmodel.py
```

* Uses best-performing model per classification task
* Computes SHAP values to identify most influential microbial phyla for disease classification 
* Generates:

  * SHAP summary plots for disease classification 
  * Final classification performance metrics



## Model Architecture and Interpretation

### `model.py` — Final Classification and SHAP Interpretation

This script performs three core classification tasks using phylum-level gut microbiota data:

---

### **Task 1: Disease Classification (Healthy vs NAFLD)**

* Objective: Identify whether the gut microbial composition can distinguish NAFLD patients from healthy individuals.
* Model Used: XGBoost
* Data Used: Combined `Healthy_male`, `Healthy_female`, `NAFLD_male`, and `NAFLD_female` sheets.
* Preprocessing:

  * Log-transformation of relative phylum abundances.
  * Class balancing via random undersampling.
* Interpretability:

  * SHAP (SHapley Additive exPlanations) used to visualize the contribution of each phylum to the classification.
  * Generates `SHAP_Disease_Classification.png` summarizing the top contributing features.

---

### Task 2: Sex Classification in Healthy Individuals

* Objective: Predict the biological sex of individuals within the healthy group using their gut microbiota profile.
* Model Used: Random Forest
* Data Used: Subset of data labeled `Healthy_male` and `Healthy_female`.
* Preprocessing:

  * Same log transformation.
  * Class balancing across sexes.
* **Output**:

  * Classification report with accuracy, precision, recall, F1-score.
  * ROC curve plotted for model evaluation.

---

### Task 3: Sex Classification in NAFLD Patients

* Objective: Assess whether gut microbiota signatures can reveal sex differences in patients already diagnosed with NAFLD.
* Model Used: XGBoost
* Data Used: Subset labeled `NAFLD_male` and `NAFLD_female`.
* Interpretability:

  * SHAP not used (for brevity and interpretability).
* Evaluation:

  * ROC and classification metrics included.

---

### SHAP Analysis Functionality

* SHAP values are calculated for the best-performing model (disease classification).
* Visualizations are saved as `.png` to provide model interpretability at the feature level.
* Bar plots rank phyla based on their mean absolute SHAP contribution.

---

### Output Summary

Each modeling task outputs:

* Textual classification report: Precision, recall, F1-score
* ROC-AUC curve
* SHAP plot (where applicable)

Example file generated:

```plaintext
SHAP_Disease_Classification_XGBoost.png
```

---

### Reproducibility

All random operations (e.g., undersampling, train/test split) are seeded with `random_state=42` to ensure reproducibility.

---
## Contact

For any questions or issues, please contact:

Kartika Sahu
kartika.sahu@niser.ac.in
National Institute of Science Education and Research, Bhubaneswar 

```

