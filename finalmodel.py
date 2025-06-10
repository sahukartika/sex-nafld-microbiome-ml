import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# --- Data Loading & Preprocessing Functions ---

def load_data(file_path, sheets):
    all_dfs = []
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df = df.set_index(df.columns[0]).T
        status, sex = sheet.split("_")
        df['Disease'] = status
        df['Sex'] = sex.lower()
        all_dfs.append(df)
    return pd.concat(all_dfs)

def normalize_log_transform(df):
    df = df.loc[df.sum(axis=1) != 0]
    X_rel = df.div(df.sum(axis=1), axis=0) + 1e-14
    return np.log(X_rel)

def balance_classes(df, label_col):
    min_size = df[label_col].value_counts().min()
    balanced = [resample(df[df[label_col] == val], n_samples=min_size, random_state=42)
                for val in df[label_col].unique()]
    return pd.concat(balanced).sample(frac=1, random_state=42)

# --- Plotting Functions ---

def plot_roc_auc(model, X_test, y_test, title):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {title}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

    return roc_auc



import matplotlib.pyplot as plt
import shap
import numpy as np
import pandas as pd
import seaborn as sns

def plot_shap(model, X, title):
    """
    Generate and save a SHAP summary bar plot with custom styling.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Compute mean absolute SHAP values per feature
    shap_val_abs = np.abs(shap_values.values).mean(axis=0)
    shap_summary = pd.Series(shap_val_abs, index=X.columns).sort_values(ascending=True)

    # Plot
    plt.figure(figsize=(8, 6))
    cmap = sns.color_palette("crest", as_cmap=True)
    colors = sns.color_palette("crest", len(shap_summary))
    plt.barh(shap_summary.index, shap_summary.values, color=colors)

    plt.xlabel("Mean |SHAP value|", fontsize=12)
    
    plt.title(f"SHAP Summary - {title}", fontsize=14, weight='bold', color='#333333')
    plt.grid(True, axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"SHAP_{title.replace(' ', '_')}.png", dpi=300)
    plt.close()



# --- Evaluation Function ---

def evaluate_model(X, y, model, task_name, use_shap=False):
    print(f"\n=== {task_name} ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    auc_score = plot_roc_auc(model, X_test, y_test, task_name)
    print(f"AUC: {auc_score:.4f}")

    
    
    if use_shap:
        plot_shap(model, X_train, task_name)

# --- Load and preprocess data ---
file_path = "phylum_abundance_by_group.xlsx"
sheets = ['Healthy_male', 'Healthy_female', 'NAFLD_male', 'NAFLD_female']
data = load_data(file_path, sheets)
X_raw = data.drop(columns=['Disease', 'Sex'])
X_log = normalize_log_transform(X_raw)

# --- Task 1: Disease Classification (WITH SHAP) ---
df_disease = X_log.copy()
df_disease['Label'] = data['Disease']
df_disease_bal = balance_classes(df_disease, 'Label')
X_disease = df_disease_bal.drop(columns='Label')
y_disease = df_disease_bal['Label'].str.lower().map({'healthy': 0, 'nafld': 1})

evaluate_model(X_disease, y_disease,
               XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
               "Disease Classification (XGBoost)",
               use_shap=True)

# --- Task 2: Sex Classification in Healthy (NO SHAP) ---
df_healthy = X_log.copy()
df_healthy['Sex'] = data['Sex'].str.lower()
df_healthy = df_healthy[data['Disease'].str.lower() == 'healthy']
df_healthy_bal = balance_classes(df_healthy, 'Sex')
X_sex_healthy = df_healthy_bal.drop(columns='Sex')
y_sex_healthy = df_healthy_bal['Sex'].map({'male': 0, 'female': 1})

evaluate_model(X_sex_healthy, y_sex_healthy,
               RandomForestClassifier(n_estimators=100, random_state=42),
               "Sex Classification in Healthy (Random Forest)",
               use_shap=False)

# --- Task 3: Sex Classification in NAFLD (NO SHAP, NO CV) ---
df_nafld = X_log.copy()
df_nafld['Sex'] = data['Sex'].str.lower()
df_nafld = df_nafld[data['Disease'].str.lower() == 'nafld']
df_nafld_bal = balance_classes(df_nafld, 'Sex')
X_sex_nafld = df_nafld_bal.drop(columns='Sex')
y_sex_nafld = df_nafld_bal['Sex'].map({'male': 0, 'female': 1})

evaluate_model(X_sex_nafld, y_sex_nafld,
               XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
               "Sex Classification in NAFLD (XGBoost)",
               use_shap=False)
