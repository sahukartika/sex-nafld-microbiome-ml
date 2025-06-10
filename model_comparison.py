import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# --- Load and Preprocess Data ---
def load_data(file_path, sheets):
    all_dfs = []
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.set_index(df.columns[0]).T
        status, sex = sheet.split("_")
        df['Disease'] = status
        df['Sex'] = sex.lower()
        all_dfs.append(df)
    data = pd.concat(all_dfs)
    return data

def normalize_log_transform(df):
    df = df.loc[df.sum(axis=1) != 0]
    X_rel = df.div(df.sum(axis=1), axis=0) + 1e-14
    return np.log(X_rel)

def balance_classes(df, label_col):
    min_size = df[label_col].value_counts().min()
    balanced = [resample(df[df[label_col] == val], n_samples=min_size, random_state=42)
                for val in df[label_col].unique()]
    return pd.concat(balanced).sample(frac=1, random_state=42)

# --- Model Evaluation ---
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'ExtraTrees': ExtraTreesClassifier(random_state=42)
    
}

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = (report['0']['f1-score'] + report['1']['f1-score']) / 2
        results[name] = f1 * 100  # convert to percentage
    return results

# --- Main ---
file_path = "phylum_abundance_by_group.xlsx"
sheets = ['Healthy_male', 'Healthy_female', 'NAFLD_male', 'NAFLD_female']
data = load_data(file_path, sheets)

X_raw = data.drop(columns=['Disease', 'Sex'])
X_log = normalize_log_transform(X_raw)

# Task 1: NAFLD vs Healthy
df_disease = X_log.copy()
df_disease['Label'] = data['Disease']
df_disease_bal = balance_classes(df_disease, 'Label')
X_disease = df_disease_bal.drop(columns='Label')
y_disease = df_disease_bal['Label'].str.lower().map({'healthy': 0, 'nafld': 1})
results_disease = evaluate_models(X_disease, y_disease)

# Task 2a: Sex classification in Healthy
df_healthy = X_log.copy()
df_healthy['Sex'] = data['Sex'].str.lower()
df_healthy = df_healthy[data['Disease'].str.lower() == 'healthy']
df_healthy_bal = balance_classes(df_healthy, 'Sex')
X_sex_healthy = df_healthy_bal.drop(columns='Sex')
y_sex_healthy = df_healthy_bal['Sex'].map({'male': 0, 'female': 1})
results_healthy_sex = evaluate_models(X_sex_healthy, y_sex_healthy)

# Task 2b: Sex classification in NAFLD
df_nafld = X_log.copy()
df_nafld['Sex'] = data['Sex'].str.lower()
df_nafld = df_nafld[data['Disease'].str.lower() == 'nafld']
df_nafld_bal = balance_classes(df_nafld, 'Sex')
X_sex_nafld = df_nafld_bal.drop(columns='Sex')
y_sex_nafld = df_nafld_bal['Sex'].map({'male': 0, 'female': 1})
results_nafld_sex = evaluate_models(X_sex_nafld, y_sex_nafld)

# --- Plotting ---
import matplotlib.pyplot as plt

tasks = ['Disease Classification', 'Sex Classificatiom in Healthy', 'Sex Classification in NAFLD']
model_names = list(models.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for tasks

# Prepare data
f1_matrix = []
for model in model_names:
    f1_matrix.append([
        results_disease.get(model, 0),
        results_healthy_sex.get(model, 0),
        results_nafld_sex.get(model, 0)
    ])
f1_matrix = np.array(f1_matrix)

# Plot
x = np.arange(len(model_names))
width = 0.2

plt.figure(figsize=(10, 6))
for i in range(3):
    plt.bar(x + i*width - width, f1_matrix[:, i], width, label=tasks[i], color=colors[i])

plt.ylabel('Average F1 Score (%)')
plt.ylim(50, 100)  # Start y-axis from 50
plt.title('Model Comparison Across Classification Tasks')
plt.xticks(x, model_names)
plt.legend()
plt.tight_layout()
plt.savefig("model_f1_score_comparison.png", dpi=300)
plt.show()
